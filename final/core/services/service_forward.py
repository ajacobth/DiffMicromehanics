"""service_forward.py — forward prediction service with process-level model cache."""
from __future__ import annotations

from core.forward import load_forward, ForwardModel

_cache: dict[str, ForwardModel] = {}


def get_model(name: str) -> ForwardModel:
    """Load and cache a ForwardModel by name ("elastic" | "thermoelastic" | "thermal").
    Subsequent calls return the cached instance.
    """
    if name not in _cache:
        _cache[name] = load_forward(name)
    return _cache[name]


def run_forward(model_name: str, inputs: dict[str, float]) -> dict[str, float]:
    """Run a forward prediction.

    Parameters
    ----------
    model_name : "elastic" | "thermoelastic" | "thermal"
    inputs     : dict in model units — must contain all model input fields.

    Returns
    -------
    outputs : dict[field_name, float] in model units.

    Raises
    ------
    KeyError  if any required input field is missing.
    """
    model = get_model(model_name)
    missing = [k for k in model.input_fields if k not in inputs]
    if missing:
        raise KeyError(f"run_forward({model_name!r}): missing input fields: {missing}")
    return model.predict(inputs)


def get_input_fields(model_name: str) -> list[str]:
    """Return the ordered list of input field names for a model."""
    return list(get_model(model_name).input_fields)


def get_output_fields(model_name: str) -> list[str]:
    """Return the ordered list of output field names for a model."""
    return list(get_model(model_name).output_fields)


def sweep_parameter(
    parameter: str,
    values: list[float],
    target_property: str = "E1",
    card_id: int = -1,
    base_inputs: dict | None = None,
) -> dict:
    """Sweep one microstructure or constituent parameter and return predictions at each value.

    Parameters
    ----------
    parameter       : ar | a11 | a22 | fiber_massfrac | matrix_modulus |
                      matrix_poisson | f_cte1 | f_cte2 | m_cte
    values          : list of values to evaluate
    target_property : output field to highlight (e.g. "E1", "CTE11")
    card_id         : base card to load inputs from (use -1 if providing base_inputs directly)
    base_inputs     : pre-built inputs dict; used when card_id == -1

    Returns
    -------
    dict with keys: parameter, target_property, rows, card_id
    """
    import core.services.service_cards as _scards

    PARAM_MAP = {
        "ar":             "ar",
        "a11":            "a11",
        "a22":            "a22",
        "fiber_massfrac": "fiber_massfrac",
        "mf":             "fiber_massfrac",
        "matrix_modulus": "matrix_modulus",
        "matrix_poisson": "matrix_poisson",
        "f_cte1":         "f_cte1",
        "f_cte2":         "f_cte2",
        "m_cte":          "m_cte",
    }
    field = PARAM_MAP.get(parameter.lower())
    if field is None:
        raise ValueError(f"Unknown parameter '{parameter}'. Choose from: {sorted(PARAM_MAP)}")

    if card_id >= 0:
        inputs_base = _scards.load_card_inputs(card_id)
    elif base_inputs is not None:
        inputs_base = dict(base_inputs)
    else:
        raise ValueError("Provide either card_id >= 0 or base_inputs.")

    rows = []
    for val in values:
        inputs = dict(inputs_base)
        inputs[field] = val
        if field in ("a11", "a22"):
            inputs["a33"] = 1.0 - inputs["a11"] - inputs["a22"]
        el_out = run_forward("elastic", inputs)
        te_out = run_forward("thermoelastic", inputs)
        rows.append({"value": val, "properties": {**el_out, **te_out}})

    return {
        "parameter":       parameter,
        "target_property": target_property.upper(),
        "rows":            rows,
        "card_id":         card_id,
    }


def warm_up_model(model_name: str) -> None:
    """Trigger JAX JIT compilation for a model by running a dummy prediction.

    Call this from a background thread after loading to avoid a stall on the
    first real prediction.
    """
    import jax.numpy as jnp
    model = get_model(model_name)
    dummy = jnp.zeros(len(model.input_fields), dtype=jnp.float32)
    model.predict_array(dummy).block_until_ready()
