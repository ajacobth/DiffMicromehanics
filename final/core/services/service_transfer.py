"""service_transfer.py — cross-printer property transfer workflow."""
from __future__ import annotations

from typing import Optional, TypedDict

import db.db as _db
from core.services.service_forward import get_model
from core.services.service_inverse import run_inverse, _ALIASES

# Constituent properties to transfer.
# (key, constituent_type, display_label, possible_db_property_names, stage_hint)
_TRANSFER_PROPS = [
    ("matrix_modulus", "polymer", "Matrix Modulus",   ["matrix_modulus"],       "Stage 1"),
    ("matrix_poisson", "polymer", "Matrix Poisson",   ["matrix_poisson"],       "Stage 1"),
    ("f_cte1",         "fiber",   "Fiber CTE α11",    ["f_cte1", "f_CTE1"],     "Stage 2"),
    ("f_cte2",         "fiber",   "Fiber CTE α22",    ["f_cte2", "f_CTE2"],     "Stage 2"),
    ("m_cte",          "polymer", "Matrix CTE",       ["m_cte", "matrix_CTE"],  "Stage 2"),
    ("k_f1",           "fiber",   "Fiber k‖",         ["k_f1"],                 "Stage 3"),
    ("k_f2",           "fiber",   "Fiber k⊥",         ["k_f2"],                 "Stage 3"),
    ("k_m",            "polymer", "Matrix k",         ["k_m"],                  "Stage 3"),
]

# Default microstructure bounds for transfer inverse
_MICRO_BOUNDS_DEFAULT: dict[str, tuple[float, float]] = {
    "a11":            (0.0,  1.0),
    "a22":            (0.0,  1.0),
    "a12":            (-0.5, 0.5),
    "a13":            (-0.5, 0.5),
    "a23":            (-0.5, 0.5),
    "fiber_massfrac": (0.01, 0.60),
    "w_f":            (0.01, 0.60),
    "ar":             (1.0,  100.0),
    "ar_f":           (1.0,  100.0),
}

# Microstructure fields that are free in transfer mode
_MICRO_FREE = frozenset(
    {"a11", "a22", "a12", "a13", "a23", "fiber_massfrac", "w_f", "ar", "ar_f"}
)


class TransferResult(TypedDict):
    source_card_id:     int
    constituent_props:  dict   # key → {value, source_tag} or None
    opt_microstructure: dict   # model units
    elastic_error:      float
    elastic_predicted:  dict
    target_outputs:     dict
    sigmas:             dict
    constituent_inputs: dict   # full input dict used for forward models
    predictions:        dict   # "elastic" | "thermoelastic" | "thermal" → dict | None


def _best_value(rows: list[dict], names: list[str]) -> Optional[dict]:
    """Return best (inferred > inputted > web, newest first) value for any of names."""
    for tag in ("inferred", "inputted", "web"):
        for r in rows:
            if r["property_name"] in names and r["source_tag"] == tag:
                return {"value": float(r["value"]), "source_tag": tag}
    return None


def resolve_constituent_props(card_id: int) -> dict[str, Optional[dict]]:
    """Resolve the 8 constituent properties from a card.
    Returns {key: {value, source_tag} or None if missing}.
    """
    card  = _db.get_print_config_card(card_id)
    frows = card["constituent_properties"]["fiber"]
    prows = card["constituent_properties"]["polymer"]
    return {
        key: _best_value(frows if ctype == "fiber" else prows, db_names)
        for key, ctype, _, db_names, _ in _TRANSFER_PROPS
    }


def build_forward_inputs(
    card_id:        int,
    microstructure: dict[str, float],
) -> dict[str, float]:
    """Assemble a complete model-unit input dict from:
      1. fiber datasheet
      2. polymer datasheet
      3. inferred constituent props (override datasheet)
      4. provided microstructure values
      5. field aliases so all three models can consume the result
    """
    card = _db.get_print_config_card(card_id)
    cfg  = card["config"]

    inputs: dict[str, float] = {}

    # datasheet
    inputs.update(_db.fiber_model_inputs(cfg["fiber_id"]))
    inputs.update(_db.polymer_model_inputs(cfg["polymer_id"]))

    # inferred constituent props (inferred > inputted)
    for ctype, cid in (("fiber", cfg["fiber_id"]), ("polymer", cfg["polymer_id"])):
        inferred: dict[str, float] = {}
        inputted: dict[str, float] = {}
        for p in _db.get_constituent_properties(ctype, cid, include_global=True):
            name = p["property_name"]
            tag  = p["source_tag"]
            if tag == "inferred" and name not in inferred:
                inferred[name] = float(p["value"])
            elif tag == "inputted" and name not in inputted:
                inputted[name] = float(p["value"])
        inputs.update(inputted)
        inputs.update(inferred)

    # microstructure
    inputs.update({k: float(v) for k, v in microstructure.items()})

    # add aliases so all three models can consume the result
    alias_pairs = [
        ("fiber_massfrac", "w_f"),
        ("ar", "ar_f"),
        ("f_cte1", "f_CTE1"),
        ("f_cte2", "f_CTE2"),
        ("m_cte", "matrix_CTE"),
    ]
    for a, b in alias_pairs:
        if a in inputs and b not in inputs:
            inputs[b] = inputs[a]
        elif b in inputs and a not in inputs:
            inputs[a] = inputs[b]

    return inputs


def run_transfer(
    source_card_id: int,
    target_outputs: dict[str, float],
    sigmas:         dict[str, float] | None = None,
    bounds:         dict[str, tuple[float, float]] | None = None,
    solver_cfg:     dict | None = None,
    manual_props:   dict[str, float] | None = None,
) -> TransferResult:
    """Full transfer workflow:
      1. Load source card + resolve constituent props
      2. Run elastic inverse (microstructure free, constituent props fixed)
      3. Run all three forward models

    Returns TransferResult — no DB writes.
    """
    constituent_props = resolve_constituent_props(source_card_id)

    # Apply any manual overrides for missing properties
    if manual_props:
        for key, value in manual_props.items():
            constituent_props[key] = {"value": float(value), "source_tag": "inputted"}

    # Build base input dict from datasheet + constituent props
    base_inputs = build_forward_inputs(source_card_id, {})

    # Determine which model fields are free (microstructure)
    elastic_model = get_model("elastic")
    free_fields  = [f for f in elastic_model.input_fields if f in _MICRO_FREE]
    fixed_fields = {f: v for f, v in base_inputs.items() if f not in _MICRO_FREE}

    # Build bounds: use defaults for microstructure fields
    effective_bounds = {**_MICRO_BOUNDS_DEFAULT}
    if bounds:
        effective_bounds.update(bounds)
    # Only keep bounds for free fields
    active_bounds = {k: effective_bounds[k] for k in free_fields if k in effective_bounds}

    result = run_inverse(
        model_name="elastic",
        fixed_inputs=fixed_fields,
        free_inputs=free_fields,
        bounds=active_bounds,
        target_outputs=target_outputs,
        sigmas=sigmas,
        solver_cfg=solver_cfg,
    )

    opt_micro = result["opt_free"]

    # Build full input dict for forward predictions
    full_inputs = build_forward_inputs(source_card_id, opt_micro)

    # Run all three forward models
    predictions: dict[str, Optional[dict]] = {}
    for model_name in ("elastic", "thermoelastic", "thermal"):
        try:
            m = get_model(model_name)
            missing = [k for k in m.input_fields if k not in full_inputs]
            if missing:
                predictions[model_name] = {"error": f"Missing inputs: {missing}"}
            else:
                predictions[model_name] = m.predict(
                    {k: full_inputs[k] for k in m.input_fields}
                )
        except FileNotFoundError:
            predictions[model_name] = None
        except Exception as exc:
            predictions[model_name] = {"error": str(exc)}

    return TransferResult(
        source_card_id=source_card_id,
        constituent_props=constituent_props,
        opt_microstructure=opt_micro,
        elastic_error=result["final_error"],
        elastic_predicted=result["predicted_outputs"],
        target_outputs=dict(target_outputs),
        sigmas=dict(sigmas or {}),
        constituent_inputs=full_inputs,
        predictions=predictions,
    )
