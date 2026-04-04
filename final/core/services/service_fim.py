"""service_fim.py — Fisher Information Matrix identifiability analysis service."""
from __future__ import annotations

import numpy as np

from core.services.service_forward import get_model


def run_fim(
    model_name:     str,
    fixed_inputs:   dict[str, float],
    free_inputs:    list[str],
    bounds:         dict[str, tuple[float, float]],
    target_outputs: dict[str, float],
    sigmas:         dict[str, float] | None = None,
    n_samples:      int = 200,
) -> dict:
    """Run Fisher Information Matrix identifiability analysis.

    Builds x_template and indices internally (extracted from gui_identifiability.py).
    Returns the full dict from core.fim.run_identifiability_check.
    """
    import jax.numpy as jnp
    from core.fim import run_identifiability_check

    model = get_model(model_name)

    # Build x_template from fixed_inputs; free vars set to midpoint of bounds
    x_np = np.zeros(len(model.input_fields), dtype=np.float32)
    for k, v in fixed_inputs.items():
        if k in model.in_idx:
            x_np[model.in_idx[k]] = float(v)
    for k in free_inputs:
        if k in bounds and k in model.in_idx:
            lo, hi = bounds[k]
            x_np[model.in_idx[k]] = (lo + hi) / 2.0
    x_template = jnp.array(x_np, dtype=jnp.float32)

    free_indices = [model.in_idx[k] for k in free_inputs if k in model.in_idx]

    sigmas_dict = sigmas or {k: 0.0 for k in target_outputs}

    return run_identifiability_check(
        model.predict_array,
        x_template,
        free_inputs,
        free_indices,
        target_outputs,
        model.out_idx,
        sigmas_dict,
        bounds,
        list(model.output_fields),
        model.output_std,
        N_samples=n_samples,
    )
