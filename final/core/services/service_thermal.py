"""service_thermal.py — thermal inverse service."""
from __future__ import annotations

from typing import Callable, Optional, TypedDict

import numpy as np

from core.services.service_forward import get_model


class ThermalResult(TypedDict):
    best_params:  object                      # ConstituentParams namedtuple
    best_loss:    float
    temperatures: list                        # list[float]
    K_pred:       dict                        # {"K11": list, "K22": list, "K33": list}


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

    K_data: dict[str, np.ndarray | None] = {}
    for col in k_cols:
        K_data[col] = (
            df[col.lower()].to_numpy(dtype=float)
            if col.lower() in df.columns else None
        )

    return temperatures, K_data


def vf_to_wf(vf: float, rho_f: float, rho_m: float) -> float:
    """Convert fiber volume fraction to mass fraction."""
    from core.inverse_thermal import vf_to_wf as _vf_to_wf
    return _vf_to_wf(vf, rho_f, rho_m)


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
    )

    K_pred = compute_composite_conductivity(
        best_params, temperatures, predictor, fixed_inputs,
    )

    return ThermalResult(
        best_params=best_params,
        best_loss=best_loss,
        temperatures=temperatures.tolist(),
        K_pred={k: (list(v) if v is not None else None) for k, v in K_pred.items()},
    )
