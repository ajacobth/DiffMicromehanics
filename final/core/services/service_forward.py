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
