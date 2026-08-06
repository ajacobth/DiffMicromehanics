"""
inverse_thermal.py
==================
Temperature-dependent thermal inverse estimation for the ``final/`` codebase.

Recovers 4 constituent thermal conductivity parameters [p1, p2, l2, t]
from measured composite conductivities K11, K22, K33 at multiple temperatures,
using the trained thermal surrogate (models/thermal/).

Constituent models (Thomas et al. 2024, Sec. 4.2):
    Polymer : K_m(T)  = p1 * sqrt(T / T_ref) + p2   (temperature-dependent)
    Fiber   : K_f1    = l2                            (temperature-independent)
              K_f2    = l2 / t                        (temperature-independent)

Usage
-----
    from forward import load_forward
    from inverse_thermal import make_batched_predictor, run_inverse_estimation

    fwd = load_forward("thermal")
    predictor = make_batched_predictor(fwd)
    best_params, best_loss = run_inverse_estimation(
        K_data, predictor, fixed_inputs,
        n_restarts=10, seed=0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# ── Reference temperature ─────────────────────────────────────────────────────
T_REF = 1.0   # °C  (Thomas et al. 2024, Sec. 4.2)

# ── Output index map (must match models/thermal/model_config.json) ────────────
_OUT_IDX = {"k11": 0, "k12": 1, "k13": 2, "k22": 3, "k23": 4, "k33": 5}

# ── Optimisation bounds [lo, hi] — physical constraints ──────────────────────
PARAM_BOUNDS = [
    (0.0,  0.05),   # p1  polymer conductivity scaling  [W/(m·°C)]  — allows k_m up to ~0.4 at 200°C
    (0.0,  0.40),   # p2  polymer conductivity offset   [W/(m·°C)]  — covers 0.05–0.4 W/m·K range
    (1.0, 20.0),    # l2  fiber longitudinal conductivity [W/(m·°C)]
    (1.01, 15.0),   # t   fiber anisotropy ratio (K_f1/K_f2)  [–]
]


# ═════════════════════════════════════════════════════════════════════════════
# Constituent conductivity models
# ═════════════════════════════════════════════════════════════════════════════

class PolymerConductivityModel:
    """
    Temperature-dependent isotropic polymer thermal conductivity.

        K_m(T) = p1 * sqrt(T / T_ref) + p2
    """

    def __init__(self, p1: float, p2: float):
        self.p1 = float(p1)
        self.p2 = float(p2)

    def __call__(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return self.p1 * np.sqrt(np.maximum(T, 0.0) / T_REF) + self.p2


class FiberConductivityModel:
    """
    Transversely isotropic fiber conductivity (temperature-independent, l1=0).

        K_f_long(T)  = l2           (constant)
        K_f_trans(T) = l2 / t       (constant)
    """

    def __init__(self, l2: float, t: float):
        self.l2 = float(l2)
        self.t  = float(t)

    def K_f_long(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return np.full_like(T, self.l2)

    def K_f_trans(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, dtype=float)
        return np.full_like(T, self.l2 / self.t)


# ═════════════════════════════════════════════════════════════════════════════
# Parameter container
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ConstituentParams:
    """
    4 free constituent parameters estimated by the inverse solver.

    p1  polymer conductivity scaling  [W/(m·°C)],       >= 0
    p2  polymer conductivity offset   [W/(m·°C)],       >= 0
    l2  fiber longitudinal conductivity [W/(m·°C)],     > 0
    t   fiber anisotropy ratio K_f1/K_f2 [–],           > 1
    """
    p1: float
    p2: float
    l2: float
    t:  float

    def to_array(self) -> np.ndarray:
        return np.array([self.p1, self.p2, self.l2, self.t], dtype=float)

    @classmethod
    def from_array(cls, x: np.ndarray) -> "ConstituentParams":
        return cls(p1=float(x[0]), p2=float(x[1]),
                   l2=float(x[2]), t=float(x[3]))


# ═════════════════════════════════════════════════════════════════════════════
# Batched predictor factory
# ═════════════════════════════════════════════════════════════════════════════

def make_batched_predictor(fwd) -> Callable:
    """
    Wrap a ForwardModel (from load_forward("thermal")) for batched evaluation.

    Parameters
    ----------
    fwd : ForwardModel
        Returned by ``load_forward("thermal")``.

    Returns
    -------
    predictor : callable
        Accepts ``X: np.ndarray, shape (N, 12)`` and returns
        ``Y: np.ndarray, shape (N, 6)``  — [k11, k12, k13, k22, k23, k33].
    """
    import jax
    import jax.numpy as jnp

    batched = jax.jit(jax.vmap(fwd.predict_array))

    def predictor(X: np.ndarray) -> np.ndarray:
        return np.asarray(batched(jnp.array(X, dtype=jnp.float32)))

    return predictor


# ═════════════════════════════════════════════════════════════════════════════
# Forward model evaluation
# ═════════════════════════════════════════════════════════════════════════════

def compute_composite_conductivity(
    params: ConstituentParams,
    temperatures: np.ndarray,
    predictor: Callable,
    fixed_inputs: Dict[str, float],
) -> np.ndarray:
    """
    Compute predicted composite conductivities at each temperature.

    Parameters
    ----------
    params        : ConstituentParams
    temperatures  : (N,) array of temperatures in °C
    predictor     : callable, (N, 12) → (N, 6)
    fixed_inputs  : dict with keys ar_f, w_f, rho_f, rho_m,
                    a11, a22, a12, a13, a23

    Returns
    -------
    K_pred : (N, 3) array — columns are [K11, K22, K33]
    """
    T = np.asarray(temperatures, dtype=float)
    N = len(T)

    poly  = PolymerConductivityModel(params.p1, params.p2)
    fiber = FiberConductivityModel(params.l2, params.t)

    X = np.column_stack([
        fiber.K_f_long(T),
        fiber.K_f_trans(T),
        poly(T),
        np.full(N, fixed_inputs["ar_f"]),
        np.full(N, fixed_inputs["w_f"]),
        np.full(N, fixed_inputs["rho_f"]),
        np.full(N, fixed_inputs["rho_m"]),
        np.full(N, fixed_inputs["a11"]),
        np.full(N, fixed_inputs["a22"]),
        np.full(N, fixed_inputs["a12"]),
        np.full(N, fixed_inputs["a13"]),
        np.full(N, fixed_inputs["a23"]),
    ]).astype(np.float32)   # (N, 12)

    Y = np.asarray(predictor(X), dtype=float)  # (N, 6)

    return np.column_stack([
        Y[:, _OUT_IDX["k11"]],
        Y[:, _OUT_IDX["k22"]],
        Y[:, _OUT_IDX["k33"]],
    ])  # (N, 3)


# ═════════════════════════════════════════════════════════════════════════════
# Objective function
# ═════════════════════════════════════════════════════════════════════════════

def objective_function(
    x: np.ndarray,
    K_data: Dict[str, Optional[Dict[str, np.ndarray]]],
    predictor: Callable,
    fixed_inputs: Dict[str, float],
) -> float:
    """
    Normalised MSE between surrogate predictions and measured data.

    Parameters
    ----------
    x            : (4,) array [p1, p2, l2, t]
    K_data       : dict {K11, K22, K33} → {"T": ndarray, "K": ndarray} or None.
                   Each channel carries its own temperature array so measurements
                   from separate test runs can be used without aligning to a
                   common temperature grid.
    predictor    : callable (N, 12) → (N, 6)
    fixed_inputs : structural parameters

    Returns
    -------
    loss : float — normalised MSE [(W/(m·°C))²]
    """
    params = ConstituentParams.from_array(x)
    loss, n_pts = 0.0, 0
    for key, col in [("K11", 0), ("K22", 1), ("K33", 2)]:
        ch = K_data.get(key)
        if ch is None:
            continue
        K_pred = compute_composite_conductivity(params, ch["T"], predictor, fixed_inputs)
        scale  = float(np.mean(ch["K"])) or 1.0   # per-channel mean → equal relative weight
        loss  += float(np.sum(((K_pred[:, col] - ch["K"]) / scale) ** 2))
        n_pts += len(ch["T"])

    return loss / max(n_pts, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Multi-start L-BFGS-B solver
# ═════════════════════════════════════════════════════════════════════════════

def run_inverse_estimation(
    K_data: Dict[str, Optional[Dict[str, np.ndarray]]],
    predictor: Callable,
    fixed_inputs: Dict[str, float],
    n_restarts: int = 10,
    seed: int = 0,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    bounds: Optional[list] = None,
) -> Tuple[ConstituentParams, float]:
    """
    Multi-start L-BFGS-B to estimate constituent thermal conductivity parameters.

    Parameters
    ----------
    K_data       : {K11, K22, K33} → {"T": ndarray, "K": ndarray} or None.
                   Each channel carries its own temperature vector.
    predictor    : callable (N, 12) → (N, 6)
    fixed_inputs : structural parameters
    n_restarts   : number of random restarts
    seed         : RNG seed
    progress_cb  : optional callback(restart, n_restarts, loss) for GUI updates
    bounds       : list of (lo, hi) per parameter [p1, p2, l2, t];
                   defaults to PARAM_BOUNDS if not supplied

    Returns
    -------
    best_params : ConstituentParams
    best_loss   : float
    """
    active_bounds = bounds if bounds is not None else PARAM_BOUNDS
    rng = np.random.default_rng(seed)
    lo  = np.array([b[0] for b in active_bounds])
    hi  = np.array([b[1] for b in active_bounds])

    best_params: Optional[ConstituentParams] = None
    best_loss   = np.inf

    for i in range(n_restarts):
        x0  = lo + rng.random(4) * (hi - lo)
        res = minimize(
            fun     = objective_function,
            x0      = x0,
            args    = (K_data, predictor, fixed_inputs),
            method  = "L-BFGS-B",
            bounds  = active_bounds,
            options = {"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
        )
        loss = float(res.fun)

        if progress_cb is not None:
            progress_cb(i + 1, n_restarts, loss)

        if loss < best_loss:
            best_loss   = loss
            best_params = ConstituentParams.from_array(res.x)

    return best_params, best_loss


# ═════════════════════════════════════════════════════════════════════════════
# Volume fraction helper
# ═════════════════════════════════════════════════════════════════════════════

def vf_to_wf(vf: float, rho_f: float, rho_m: float) -> float:
    """Convert fiber volume fraction to weight fraction."""
    return (vf * rho_f) / ((1.0 - vf) * rho_m + vf * rho_f)
