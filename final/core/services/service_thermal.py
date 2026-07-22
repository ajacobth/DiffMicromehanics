"""service_thermal.py — thermal inverse service."""
from __future__ import annotations

from typing import Callable, Optional, TypedDict

import numpy as np

from core.services.service_forward import get_model

# Canonical bounds for the 4 free parameters [p1, p2, l2, t].
# Changing these here propagates to both the agent and the GUI.
THERMAL_BOUNDS = [
    (0.0,  0.05),   # p1  polymer conductivity scaling  [W/m·K]
    (0.0,  0.40),   # p2  polymer conductivity offset   [W/m·K]
    (1.0, 20.0),    # l2  fiber longitudinal conductivity [W/m·K]
    (1.01, 15.0),   # t   fiber anisotropy ratio [-]
]


class ThermalResult(TypedDict):
    p1:           float   # polymer conductivity scaling [W/m·K]
    p2:           float   # polymer conductivity offset  [W/m·K]
    l2:           float   # fiber longitudinal conductivity [W/m·K]
    t:            float   # fiber anisotropy ratio (K_f_long / K_f_trans) [-]
    best_loss:    float
    temperatures: list    # list[float]
    K_pred:       dict    # {"K11": list[float], "K22": list[float], "K33": list[float]}


def load_thermal_data(
    path: str,
    temperature_col: str = "temperature",
    k_cols: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray | None]]:
    """Load temperature-dependent conductivity data from CSV or Excel.

    Returns (temperatures, K_data) where K_data is {"K11": arr|None, ...}.
    """
    import os
    import pandas as pd

    k_cols = k_cols or ["K11", "K22", "K33"]
    ext = os.path.splitext(path)[1].lower()
    df  = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    tcol = temperature_col.lower()
    if tcol not in df.columns:
        raise ValueError(
            f"Temperature column '{temperature_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    temperatures = df[tcol].to_numpy(dtype=float)

    # Column aliases: K11 → ["k11", "k11_wmk"], K22 → ["k22", "k22_wmk"], etc.
    _K_ALIASES = {k: [k.lower(), f"{k.lower()}_wmk"] for k in k_cols}

    K_data: dict[str, np.ndarray | None] = {}
    for col in k_cols:
        matched = next((a for a in _K_ALIASES[col] if a in df.columns), None)
        K_data[col] = df[matched].to_numpy(dtype=float) if matched else None

    return temperatures, K_data


def vf_to_wf(vf: float, rho_f: float, rho_m: float) -> float:
    """Convert fiber volume fraction to mass fraction."""
    from core.inverse_thermal import vf_to_wf as _vf_to_wf
    return _vf_to_wf(vf, rho_f, rho_m)


def compute_conductivity_curves(
    p1: float,
    p2: float,
    l2: float,
    t: float,
    temperatures: "np.ndarray | list",
) -> dict:
    """Compute constituent conductivity curves over a temperature array.

    Returns a dict with float lists: "k_polymer", "k_fiber_long", "k_fiber_trans".
    """
    from core.inverse_thermal import PolymerConductivityModel, FiberConductivityModel
    T = np.asarray(temperatures, dtype=float)
    poly  = PolymerConductivityModel(p1, p2)
    fiber = FiberConductivityModel(l2, t)
    return {
        "k_polymer":     poly(T).tolist(),
        "k_fiber_long":  fiber.K_f_long(T).tolist(),
        "k_fiber_trans": fiber.K_f_trans(T).tolist(),
    }


def run_thermal_inverse(
    fixed_inputs: dict[str, float],
    temperatures: np.ndarray,
    K_data: dict[str, np.ndarray | None],
    n_restarts: int = 5,
    seed: int = 42,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> ThermalResult:
    """Run the thermal inverse estimation.

    progress_cb(restart_idx, n_restarts, current_loss) — optional callback for UI updates.
    """
    from core.inverse_thermal import (
        make_batched_predictor,
        compute_composite_conductivity,
        run_inverse_estimation,
    )

    fwd_model  = get_model("thermal")
    predictor  = make_batched_predictor(fwd_model)

    best_params, best_loss = run_inverse_estimation(
        temperatures=temperatures,
        K_data=K_data,
        predictor=predictor,
        fixed_inputs=fixed_inputs,
        n_restarts=n_restarts,
        seed=seed,
        progress_cb=progress_cb,
        bounds=THERMAL_BOUNDS,
    )

    K_pred_arr = compute_composite_conductivity(
        best_params, temperatures, predictor, fixed_inputs,
    )  # (N, 3) ndarray — columns are [K11, K22, K33]

    return ThermalResult(
        p1=float(best_params.p1),
        p2=float(best_params.p2),
        l2=float(best_params.l2),
        t=float(best_params.t),
        best_loss=best_loss,
        temperatures=temperatures.tolist(),
        K_pred={
            "K11": K_pred_arr[:, 0].tolist(),
            "K22": K_pred_arr[:, 1].tolist(),
            "K33": K_pred_arr[:, 2].tolist(),
        },
    )
