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
) -> dict[str, dict[str, np.ndarray] | None]:
    """Load temperature-dependent conductivity data from CSV or Excel.

    Returns K_data: {"K11": {"T": arr, "K": arr} | None, "K22": ..., "K33": ...}.

    Two CSV layouts are accepted:

    1. Shared temperature column (legacy):
         Temperature, K11_WmK, K22_WmK, K33_WmK
         25.6,        1.43,    0.557,    0.294

    2. Per-channel temperature columns (preferred when each direction was
       measured on a separate run at slightly different temperatures):
         T_K11, K11_WmK, T_K22, K22_WmK, T_K33, K33_WmK
         25.6,  1.43,    26.1,  0.557,   25.8,  0.294

    The per-channel layout is detected automatically when any column named
    t_k11, t_k22, or t_k33 is found.  Rows with NaN in a channel's own T or K
    column are silently dropped for that channel only, so channels can have
    different numbers of data points.
    """
    import os
    import pandas as pd

    k_cols = k_cols or ["K11", "K22", "K33"]
    ext = os.path.splitext(path)[1].lower()
    df  = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(how="all").reset_index(drop=True)

    # Column alias maps
    _K_ALIASES = {k: [k.lower(), f"{k.lower()}_wmk"] for k in k_cols}
    _T_ALIASES = {k: [f"t_{k.lower()}", f"t_{k.lower()}_c", f"temp_{k.lower()}"]
                  for k in k_cols}

    # Detect per-channel layout: any t_k11 / t_k22 / t_k33 column present?
    has_per_channel = any(
        any(a in df.columns for a in _T_ALIASES[k]) for k in k_cols
    )

    K_data: dict[str, dict[str, np.ndarray] | None] = {}

    if has_per_channel:
        for col in k_cols:
            t_col = next((a for a in _T_ALIASES[col] if a in df.columns), None)
            k_col = next((a for a in _K_ALIASES[col] if a in df.columns), None)
            if t_col and k_col:
                mask = df[[t_col, k_col]].notna().all(axis=1)
                K_data[col] = {
                    "T": df.loc[mask, t_col].to_numpy(dtype=float),
                    "K": df.loc[mask, k_col].to_numpy(dtype=float),
                }
            else:
                K_data[col] = None
    else:
        # Shared temperature column — legacy format
        tcol = temperature_col.lower()
        if tcol not in df.columns:
            raise ValueError(
                f"Temperature column '{temperature_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        temperatures = df[tcol].to_numpy(dtype=float)
        for col in k_cols:
            matched = next((a for a in _K_ALIASES[col] if a in df.columns), None)
            if matched:
                K_data[col] = {
                    "T": temperatures.copy(),
                    "K": df[matched].to_numpy(dtype=float),
                }
            else:
                K_data[col] = None

    return K_data


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
    K_data: dict[str, dict[str, np.ndarray] | None],
    n_restarts: int = 5,
    seed: int = 42,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
) -> ThermalResult:
    """Run the thermal inverse estimation.

    K_data: {"K11": {"T": arr, "K": arr} | None, ...} — per-channel temperatures.
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
        K_data=K_data,
        predictor=predictor,
        fixed_inputs=fixed_inputs,
        n_restarts=n_restarts,
        seed=seed,
        progress_cb=progress_cb,
        bounds=THERMAL_BOUNDS,
    )

    # Build a sorted evaluation grid from the union of all channel temperatures
    # so K_pred curves span the full measured range.
    all_T = np.sort(np.unique(np.concatenate([
        ch["T"] for ch in K_data.values() if ch is not None
    ])))

    K_pred_arr = compute_composite_conductivity(
        best_params, all_T, predictor, fixed_inputs,
    )  # (N, 3) ndarray — columns are [K11, K22, K33]

    return ThermalResult(
        p1=float(best_params.p1),
        p2=float(best_params.p2),
        l2=float(best_params.l2),
        t=float(best_params.t),
        best_loss=best_loss,
        temperatures=all_T.tolist(),
        K_pred={
            "K11": K_pred_arr[:, 0].tolist(),
            "K22": K_pred_arr[:, 1].tolist(),
            "K33": K_pred_arr[:, 2].tolist(),
        },
    )
