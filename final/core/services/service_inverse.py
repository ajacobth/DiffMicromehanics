"""service_inverse.py — elastic / thermoelastic inverse solver service."""
from __future__ import annotations

import numpy as np
from typing import Optional, TypedDict

from core.services.service_forward import get_model

# Field aliases — multiple model variants for the same physical quantity.
# This is the single canonical resolution point for these aliases.
_ALIASES: dict[str, str] = {
    "fiber_massfrac": "w_f",
    "w_f":            "fiber_massfrac",
    "ar":             "ar_f",
    "ar_f":           "ar",
    "f_cte1":         "f_CTE1",
    "f_CTE1":         "f_cte1",
    "f_cte2":         "f_CTE2",
    "f_CTE2":         "f_cte2",
    "m_cte":          "matrix_CTE",
    "matrix_CTE":     "m_cte",
}


class InverseResult(TypedDict):
    model:               str
    opt_free:            dict   # field -> float, model units
    predicted_outputs:   dict   # all output fields, model units
    target_outputs:      dict   # targets only, model units
    final_error:         float
    solver_cfg:          dict
    orientation_warning: Optional[str]  # set if the solved a_ij is not PSD


def run_inverse(
    model_name:     str,
    fixed_inputs:   dict[str, float],
    free_inputs:    list[str],
    bounds:         dict[str, tuple[float, float]] | None,
    target_outputs: dict[str, float],
    sigmas:         dict[str, float] | None = None,
    solver_cfg:     dict | None = None,
    init_vals:      list[float] | None = None,
) -> InverseResult:
    """Run the inverse solver.

    - Loads (or retrieves cached) model for model_name.
    - Applies orientation sum constraint automatically if a11/a22 are free.
    - Uses "lbfgsb" method by default.
    - All values in model units.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from core.inverse import (
        InverseProblem,
        _solve,
        _assemble_x,
        make_orientation_sum_constraint,
    )

    cfg = {
        "method": "lbfgsb",
        "constraint_penalty": 1e4,
        "use_epsilon_loss": False,
        "epsilon_scale": 1.0,
        "maxiter": 300,
        "tol": 1e-9,
        "seed": 42,
    }
    if solver_cfg:
        cfg.update(solver_cfg)

    model = get_model(model_name)

    # Drop any fixed_inputs whose keys aren't in this model (e.g. rho_f for elastic)
    fixed_inputs = {k: v for k, v in fixed_inputs.items() if k in model.in_idx}

    constraints = []
    c = make_orientation_sum_constraint(free_inputs)
    if c is not None:
        constraints.append(c)

    prob = InverseProblem(
        fixed_inputs=fixed_inputs,
        free_inputs=free_inputs,
        target_outputs=target_outputs,
        constraints=tuple(constraints),
    )

    epsilon_scale = float(cfg.get("epsilon_scale", 1.0))
    sigmas_list = [
        float((sigmas or {}).get(k, 0.0)) * epsilon_scale
        for k in target_outputs.keys()
    ]

    # Initial guess: use caller-supplied values when provided; fall back to
    # midpoint of bounds, then fixed_inputs value, then 0.
    if init_vals is not None and len(init_vals) == len(free_inputs):
        computed_init = list(init_vals)
    else:
        computed_init = []
        for k in free_inputs:
            if bounds and k in bounds:
                lo, hi = bounds[k]
                computed_init.append((lo + hi) / 2.0)
            elif k in fixed_inputs:
                computed_init.append(float(fixed_inputs[k]))
            else:
                computed_init.append(0.0)
    init64 = jnp.array(computed_init, jnp.float64)

    free_vec, final_err = _solve(
        model.predict_array, prob,
        model.in_idx, model.out_idx, len(model.input_fields),
        model.output_std,
        init64, bounds,
        method=cfg["method"],
        penalty=float(cfg["constraint_penalty"]),
        sigmas_list=sigmas_list,
        use_eps_loss=bool(cfg.get("use_epsilon_loss", False)),
        maxiter=int(cfg["maxiter"]),
        tol=float(cfg["tol"]),
        seed=int(cfg.get("seed", 42)),
    )

    x_star = _assemble_x(free_vec, prob, model.in_idx, len(model.input_fields))
    y_star = model.predict_array(x_star)
    y_np   = np.asarray(y_star)

    opt_free = {k: float(v) for k, v in zip(free_inputs, free_vec)}
    predicted_outputs = {k: float(y_np[model.out_idx[k]]) for k in model.output_fields}

    # Orientation tensor may have been (partially) free — re-check PSD on the
    # solved values merged with whatever was fixed.
    orientation_warning = validate_orientation_tensor({**fixed_inputs, **opt_free})

    return InverseResult(
        model=model_name,
        opt_free=opt_free,
        predicted_outputs=predicted_outputs,
        target_outputs=dict(target_outputs),
        final_error=final_err,
        solver_cfg=cfg,
        orientation_warning=orientation_warning,
    )


def validate_orientation_tensor(inputs: dict[str, float]) -> str | None:
    """Check that the orientation tensor in inputs is positive semi-definite.
    Returns None if valid, or an error message string if not.
    """
    try:
        import numpy as np
        a11 = float(inputs.get("a11", 0.0))
        a22 = float(inputs.get("a22", 0.0))
        a12 = float(inputs.get("a12", 0.0))
        a13 = float(inputs.get("a13", 0.0))
        a23 = float(inputs.get("a23", 0.0))
        a33 = 1.0 - a11 - a22

        A = np.array([
            [a11, a12, a13],
            [a12, a22, a23],
            [a13, a23, a33],
        ])
        eigvals = np.linalg.eigvalsh(A)
        if np.any(eigvals < -1e-6):
            return (
                f"Orientation tensor is not positive semi-definite "
                f"(min eigenvalue = {eigvals.min():.4f})"
            )
    except Exception as exc:
        return f"Orientation tensor validation error: {exc}"
    return None


def assemble_full_inputs(
    model_name:   str,
    fixed_inputs: dict[str, float],
    free_values:  dict[str, float],
) -> dict[str, float]:
    """Merge fixed + free values into a complete model input dict.

    Handles field-name aliases (fiber_massfrac/w_f, ar/ar_f, f_cte1/f_CTE1, etc.)
    so callers never need to know which variant a model uses.
    """
    model = get_model(model_name)
    combined = {**fixed_inputs, **free_values}

    result: dict[str, float] = {}
    for field in model.input_fields:
        if field in combined:
            result[field] = float(combined[field])
        elif field in _ALIASES and _ALIASES[field] in combined:
            result[field] = float(combined[_ALIASES[field]])

    return result
