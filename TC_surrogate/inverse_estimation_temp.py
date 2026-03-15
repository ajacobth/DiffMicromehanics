"""
inverse_estimation_temp.py
==========================
Gradient-based inverse estimation of constituent thermal conductivities as
functions of temperature.

Given measured composite conductivities K11, K22, K33 at multiple temperatures,
estimates the 4 constituent parameters [p1, p2, l2, t] by minimising MSE
between NN-surrogate predictions and measured data.

Assumption: fiber conductivity is temperature-independent.
  K_f_long(T) = l2   (constant)
  K_f_trans(T) = l2 / t  (constant)

Only the polymer/matrix conductivity varies with temperature:
  K_m(T) = p1 * sqrt(T / T_ref) + p2

Method follows Thomas et al. (2024), Comput. Methods Appl. Mech. Eng., Sec. 4.2,
replacing Bayesian MCMC with deterministic multi-start L-BFGS-B optimisation.

The trained NN surrogate is used as the forward model: at each temperature the
constituent conductivities produced by the parameterisations are assembled into
the NN input vector, and the NN returns K11, K22, K33 for that temperature.

No file I/O or user interaction in this module.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from scipy.optimize import minimize

# ─── Reference temperature (Thomas et al. 2024, Sec. 4.2) ────────────────────
T_REF = 1.0  # °C — fixed reference temperature for polymer parameterisation

# ─── Input field order — must match TC surrogate training data ────────────────
INPUT_FIELD_NAMES = [
    "k_f1", "k_f2", "k_m", "ar_f",
    "w_f", "rho_f", "rho_m",
    "a11", "a22", "a12", "a13", "a23",   # a33 = 1 − a11 − a22 (implicit)
]

# ─── Output field order — must match TC surrogate training data ───────────────
OUTPUT_FIELD_NAMES = ["k11", "k12", "k13", "k22", "k23", "k33"]
OUT_IDX = {n: i for i, n in enumerate(OUTPUT_FIELD_NAMES)}

# ─── Bounds for L-BFGS-B (physical constraints, Thomas et al. 2024) ──────────
# Fiber conductivity is assumed temperature-independent; l1 is fixed at 0.
# Only 4 parameters are optimised: [p1, p2, l2, t].
PARAM_BOUNDS = [
    (0.0,   7e-3),    # p1: polymer conductivity scaling      [W/(m·°C)]
    (0.0,   8e-2),    # p2: polymer conductivity offset        [W/(m·°C)]
    (1.0, 20.0),    # l2: fiber long. conductivity           [W/(m·°C)]
    (1.01, 6.0),    # t:  fiber anisotropy ratio (long/trans) [dimensionless]
]


# =============================================================================
# Constituent property models  (Thomas et al. 2024, Section 4.2)
# =============================================================================

class PolymerConductivityModel:
    """
    Temperature-dependent isotropic polymer (matrix) thermal conductivity.

    Functional form (Thomas et al. 2024, Sec. 4.2):

        K_p(T) = p1 * sqrt(T / T_ref) + p2

    Parameters
    ----------
    p1 : float
        Conductivity scaling coefficient (>= 0)  [W/(m·°C)]
    p2 : float
        Conductivity offset (>= 0)               [W/(m·°C)]
    """

    def __init__(self, p1: float, p2: float):
        self.p1 = float(p1)
        self.p2 = float(p2)

    def __call__(self, T: np.ndarray) -> np.ndarray:
        """
        Evaluate polymer conductivity at temperature(s) T.

        Parameters
        ----------
        T : array_like
            Temperature(s) in °C.  Should be > 0 for a physically valid sqrt.

        Returns
        -------
        K_p : np.ndarray
            Polymer thermal conductivity [W/(m·°C)], same shape as T.
        """
        T = np.asarray(T, dtype=float)
        return self.p1 * np.sqrt(np.maximum(T, 0.0) / T_REF) + self.p2


class FiberConductivityModel:
    """
    Temperature-dependent transversely isotropic fiber thermal conductivity.

    Longitudinal (Thomas et al. 2024, Sec. 4.2):
        K_f_long(T) = l1 * T + l2

    Transverse (anisotropy ratio t = K_f_long / K_f_trans, must be > 1
    for carbon fiber where transverse < longitudinal):
        K_f_trans(T) = K_f_long(T) / t

    Parameters
    ----------
    l1 : float
        Temperature slope of longitudinal conductivity [W/(m·°C²)],  >= 0
    l2 : float
        Room-temperature longitudinal conductivity intercept [W/(m·°C)], > 0
    t  : float
        Anisotropy ratio K_f_long / K_f_trans  (> 1 for carbon fiber)
    """

    def __init__(self, l1: float, l2: float, t: float):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.t  = float(t)

    def K_f_long(self, T: np.ndarray) -> np.ndarray:
        """
        Longitudinal fiber conductivity [W/(m·°C)] at temperature(s) T.

        Parameters
        ----------
        T : array_like — temperature in °C

        Returns
        -------
        K_fl : np.ndarray, same shape as T
        """
        T = np.asarray(T, dtype=float)
        return self.l1 * T + self.l2

    def K_f_trans(self, T: np.ndarray) -> np.ndarray:
        """
        Transverse fiber conductivity [W/(m·°C)] at temperature(s) T.

        Parameters
        ----------
        T : array_like — temperature in °C

        Returns
        -------
        K_ft : np.ndarray, same shape as T
        """
        return self.K_f_long(T) / self.t


# =============================================================================
# Parameter container
# =============================================================================

@dataclass
class ConstituentParams:
    """
    Container for the 4 scalar constituent parameters to be estimated.

    Fiber conductivity is assumed temperature-independent (l1 fixed at 0).

    Attributes
    ----------
    p1 : float   Polymer conductivity scaling  [W/(m·°C)],      >= 0
    p2 : float   Polymer conductivity offset   [W/(m·°C)],      >= 0
    l1 : float   Fiber long. temp. slope — FIXED at 0.0 (temperature-independent)
    l2 : float   Fiber long. conductivity     [W/(m·°C)],       > 0
    t  : float   Fiber anisotropy ratio       [dimensionless],  > 1
    """
    p1: float
    p2: float
    l2: float
    t:  float
    l1: float = 0.0  # fixed — fiber conductivity does not vary with temperature

    def to_array(self) -> np.ndarray:
        """Pack the 4 free parameters into a 1-D numpy array [p1, p2, l2, t]."""
        return np.array([self.p1, self.p2, self.l2, self.t], dtype=float)

    @classmethod
    def from_array(cls, x: np.ndarray) -> "ConstituentParams":
        """Unpack from a 1-D array [p1, p2, l2, t]; l1 is fixed at 0."""
        return cls(
            p1=float(x[0]), p2=float(x[1]),
            l2=float(x[2]), t=float(x[3]),
        )


# =============================================================================
# Forward model — NN surrogate
# =============================================================================

def compute_composite_conductivity(
    params: ConstituentParams,
    temperatures: np.ndarray,
    nn_predictor: Callable,
    fixed_inputs: Dict[str, float],
) -> np.ndarray:
    """
    Compute predicted composite conductivities [K11, K22, K33] at each
    temperature by evaluating the trained NN surrogate.

    For each temperature T:
      1. Constituent conductivities are computed from the parameterisations:
             k_f1    = l2                    (fiber longitudinal — constant)
             k_f2    = l2 / t               (fiber transverse  — constant)
             k_m(T)  = p1*sqrt(T) + p2      (polymer/matrix — temperature-dependent)
      2. These are assembled into the NN input vector together with the
         fixed structural parameters (aspect ratio, weight fraction, densities,
         orientation tensor).
      3. The NN returns all 6 output components; K11, K22, K33 are extracted.

    Parameters
    ----------
    params        : ConstituentParams — the 5 constituent parameters
    temperatures  : np.ndarray, shape (N,) — temperatures in °C
    nn_predictor  : callable — NN forward pass; accepts (N, 12) array-like and
                    returns (N, 6) array-like: [k11, k12, k13, k22, k23, k33]
    fixed_inputs  : dict — structural parameters constant across temperatures:
                    keys: 'ar_f', 'w_f', 'rho_f', 'rho_m',
                          'a11', 'a22', 'a12', 'a13', 'a23'

    Returns
    -------
    K_pred : np.ndarray, shape (N, 3)
        Columns are [K11, K22, K33] in W/(m·°C).
    """
    T = np.asarray(temperatures, dtype=float)
    N = len(T)

    poly_model  = PolymerConductivityModel(params.p1, params.p2)
    fiber_model = FiberConductivityModel(params.l1, params.l2, params.t)

    k_m  = poly_model(T)             # (N,) — polymer/matrix conductivity
    k_f1 = fiber_model.K_f_long(T)   # (N,) — fiber longitudinal
    k_f2 = fiber_model.K_f_trans(T)  # (N,) — fiber transverse

    # Assemble NN input matrix (N, 12); column order = INPUT_FIELD_NAMES
    X = np.column_stack([
        k_f1,
        k_f2,
        k_m,
        np.full(N, fixed_inputs["ar_f"]),
        np.full(N, fixed_inputs["w_f"]),
        np.full(N, fixed_inputs["rho_f"]),
        np.full(N, fixed_inputs["rho_m"]),
        np.full(N, fixed_inputs["a11"]),
        np.full(N, fixed_inputs["a22"]),
        np.full(N, fixed_inputs["a12"]),
        np.full(N, fixed_inputs["a13"]),
        np.full(N, fixed_inputs["a23"]),
    ]).astype(np.float32)  # shape (N, 12)

    # NN forward pass — output shape (N, 6): [k11, k12, k13, k22, k23, k33]
    Y = np.array(nn_predictor(X), dtype=float)

    # Extract the three diagonal composite conductivity components
    K11 = Y[:, OUT_IDX["k11"]]
    K22 = Y[:, OUT_IDX["k22"]]
    K33 = Y[:, OUT_IDX["k33"]]

    return np.column_stack([K11, K22, K33])  # (N, 3)


# =============================================================================
# Objective function
# =============================================================================

def objective_function(
    x: np.ndarray,
    temperatures: np.ndarray,
    K_data: Dict[str, Optional[np.ndarray]],
    nn_predictor: Callable,
    fixed_inputs: Dict[str, float],
) -> float:
    """
    MSE between NN-surrogate composite conductivity and measured data.

    Fiber conductivity is temperature-independent (l1 fixed at 0); only the
    polymer conductivity varies with temperature.

    Parameters
    ----------
    x            : np.ndarray, shape (4,) — [p1, p2, l2, t]
    temperatures : np.ndarray, shape (N,) — temperatures in °C
    K_data       : dict — measured composite conductivities (None if absent)
    nn_predictor : callable — NN forward pass
    fixed_inputs : dict — structural parameters

    Returns
    -------
    loss : float — normalised MSE  [(W/(m·°C))²]
    """
    params = ConstituentParams.from_array(x)
    K_pred = compute_composite_conductivity(
        params, temperatures, nn_predictor, fixed_inputs
    )
    # K_pred columns: [K11_pred (0), K22_pred (1), K33_pred (2)]

    directions = [("K11", 0), ("K22", 1), ("K33", 2)]
    loss     = 0.0
    n_points = 0

    for key, col in directions:
        if K_data.get(key) is not None:
            residuals = K_pred[:, col] - K_data[key]
            loss     += float(np.sum(residuals ** 2))
            n_points += len(temperatures)

    return loss / max(n_points, 1)


# =============================================================================
# Multi-start L-BFGS-B optimiser
# =============================================================================

def run_inverse_estimation(
    temperatures: np.ndarray,
    K_data: Dict[str, Optional[np.ndarray]],
    nn_predictor: Callable,
    fixed_inputs: Dict[str, float],
    n_restarts: int = 10,
    seed: int = 0,
) -> Tuple[ConstituentParams, float]:
    """
    Multi-start L-BFGS-B optimisation to recover constituent parameters.

    Optimises 4 parameters [p1, p2, l2, t].  Fiber conductivity is fixed as
    temperature-independent (l1 = 0); only the polymer conductivity varies
    with temperature.

    Each restart draws a random initial point uniformly within PARAM_BOUNDS,
    runs scipy's L-BFGS-B, and the solution with the lowest final MSE is kept.

    Parameters
    ----------
    temperatures : np.ndarray, shape (N,) — temperatures in °C
    K_data       : dict — measured composite conductivities (None if unavailable)
    nn_predictor : callable — NN forward pass
    fixed_inputs : dict — structural parameters constant across temperatures
    n_restarts   : int — number of independent random restarts (>= 1)
    seed         : int — random seed for reproducibility

    Returns
    -------
    best_params : ConstituentParams — estimated constituent parameters
    best_loss   : float — final objective value at the best solution
    """
    rng = np.random.default_rng(seed)

    lo = np.array([b[0] for b in PARAM_BOUNDS])
    hi = np.array([b[1] for b in PARAM_BOUNDS])

    best_params: Optional[ConstituentParams] = None
    best_loss   = np.inf

    print("  Fiber conductivity fixed as temperature-independent (l1 = 0).")
    print("  Optimising 4 parameters: [p1, p2, l2, t]")

    for restart in range(n_restarts):
        # Uniform random initial point within physical bounds
        x0 = lo + rng.random(len(PARAM_BOUNDS)) * (hi - lo)

        result = minimize(
            fun=objective_function,
            x0=x0,
            args=(temperatures, K_data, nn_predictor, fixed_inputs),
            method="L-BFGS-B",
            bounds=PARAM_BOUNDS,
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
        )

        loss = float(result.fun)
        print(f"  Restart {restart + 1:2d}/{n_restarts}:  loss = {loss:.6e}"
              f"  {'*best*' if loss < best_loss else ''}")

        if loss < best_loss:
            best_loss   = loss
            best_params = ConstituentParams.from_array(result.x)

    return best_params, best_loss
