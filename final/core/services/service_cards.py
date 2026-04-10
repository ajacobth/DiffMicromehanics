"""service_cards.py — material card CRUD (save / load results)."""
from __future__ import annotations

from typing import Optional

import db.db as _db

# ── field classification ───────────────────────────────────────────────────────
# These constants define how model input fields map to DB tables.
# They live here — not in gui_card_dialogs.py — because they are business logic.

_MICRO_DB_MAP: dict[str, str] = {
    "a11": "a11", "a22": "a22", "a12": "a12", "a13": "a13", "a23": "a23",
    "ar":  "ar",  "ar_f": "ar",
    "fiber_massfrac": "mf", "w_f": "mf",
}
_MICRO_FIELDS = frozenset(_MICRO_DB_MAP)

_FIBER_FIELDS = frozenset({
    "e1", "e2", "g12", "f_nu12", "f_nu23", "fiber_density", "rho_f",
    "f_CTE1", "f_CTE2",
    "f_cte1", "f_cte2",
    "k_f1", "k_f2",
})

_POLYMER_FIELDS = frozenset({
    "matrix_modulus", "matrix_poisson", "matrix_density", "rho_m",
    "matrix_CTE",
    "m_cte",
    "k_m",
})

# DB microstructure field → possible model field names (for loading back)
_DB_TO_MODEL: dict[str, list[str]] = {
    "mf":  ["fiber_massfrac", "w_f"],
    "ar":  ["ar", "ar_f"],
    "a11": ["a11"], "a22": ["a22"], "a12": ["a12"],
    "a13": ["a13"], "a23": ["a23"],
}

# Thermoelastic model outputs both mechanical and CTE fields.
# Only the CTE fields should be saved to composite_property_values;
# mechanical properties (E, G, nu) must come from the elastic model only.
_TE_CTE_OUTPUTS = frozenset({
    "CTE11", "CTE22", "CTE33", "CTE12", "CTE13", "CTE23",
})


# ── load ───────────────────────────────────────────────────────────────────────

def load_card(print_config_id: int) -> dict:
    """Full card dict. Wraps db.get_print_config_card."""
    return _db.get_print_config_card(print_config_id)


def get_latest_microstructure(print_config_id: int) -> Optional[dict]:
    """Latest microstructure snapshot for a card, or None if none recorded."""
    return _db.get_latest_microstructure(print_config_id)


def load_card_inputs(print_config_id: int) -> dict[str, float]:
    """Resolve a card into a flat model-unit input dict.

    Resolution order per field:
      1. inferred constituent_property_values
      2. inputted constituent_property_values
      3. latest microstructure snapshot
      4. datasheet (fibers / polymers tables)
    """
    card = _db.get_print_config_card(print_config_id)
    cfg  = card["config"]

    # Start from datasheet values
    result: dict[str, float] = {}
    result.update(_db.fiber_model_inputs(cfg["fiber_id"]))
    result.update(_db.polymer_model_inputs(cfg["polymer_id"]))

    # Overlay microstructure snapshot (inputted provenance)
    micro = card.get("microstructure")
    if micro:
        for db_field, model_fields in _DB_TO_MODEL.items():
            val = micro.get(db_field)
            if val is not None:
                for mf in model_fields:
                    result[mf] = float(val)

    # Overlay constituent properties: inputted then inferred (inferred wins)
    def _apply_constituent(ctype: str, cid: int):
        inferred: dict[str, float] = {}
        inputted: dict[str, float] = {}
        for p in _db.get_constituent_properties(ctype, cid, include_global=True):
            name = p["property_name"]
            tag  = p["source_tag"]
            if tag == "inferred" and name not in inferred:
                inferred[name] = float(p["value"])
            elif tag == "inputted" and name not in inputted:
                inputted[name] = float(p["value"])
        result.update(inputted)
        result.update(inferred)

    _apply_constituent("fiber",   cfg["fiber_id"])
    _apply_constituent("polymer", cfg["polymer_id"])

    return result


# ── save — elastic / thermoelastic inverse ─────────────────────────────────────

def save_inverse_result(
    result:     dict,
    fiber_id:   int,
    polymer_id: int,
    printer_id: Optional[int] = None,
    card_id:    Optional[int] = None,
    card_name:  str = "",
    notes:      str = "",
) -> int:
    """Persist an elastic or thermoelastic inverse result.

    `result` must have:
        model, free_variables (or opt_free), fixed_inputs,
        predicted_outputs, target_outputs, sigmas, final_optimiser_error (or final_error),
        solver (or solver_cfg)

    Creates or updates the print_config if needed, then writes:
      - microstructure_snapshot
      - constituent_property_values (free vars → inferred, fixed vars → inputted)
      - inference_run
      - experimental_measurements (targets)
      - composite_property_values (all predicted outputs)

    Returns print_config_id.
    """
    free     = result.get("free_variables") or result.get("opt_free", {})
    fixed    = result.get("fixed_inputs", {})
    all_in   = {**fixed, **free}
    outputs  = result.get("predicted_outputs", {})
    targets  = result.get("target_outputs", {})
    sigmas   = result.get("sigmas", {}) or {}
    model_nm = result.get("model", "unknown")
    loss     = float(result.get("final_optimiser_error") or result.get("final_error", 0.0))
    solver   = result.get("solver") or result.get("solver_cfg", {})

    # 1. Get or create print config
    if card_id is None:
        name = card_name or f"{model_nm} card"
        cfg_id = _db.create_print_config(
            name=name, fiber_id=fiber_id, polymer_id=polymer_id,
            printer_id=printer_id, notes=notes,
        )
    else:
        cfg_id = card_id

    # 2. Microstructure snapshot
    micro_vals: dict[str, float | None] = {}
    micro_prov: dict[str, str] = {}
    for src_field, db_field in _MICRO_DB_MAP.items():
        if src_field in all_in:
            micro_vals[db_field] = all_in[src_field]
            micro_prov[db_field] = "inferred" if src_field in free else "inputted"

    snap_id: Optional[int] = None
    if micro_vals:
        _SNAP_FIELDS = ("mf", "ar", "a11", "a22", "a12", "a13", "a23")
        latest = _db.get_latest_microstructure(cfg_id)
        has_inferred_micro = any(v == "inferred" for v in micro_prov.values())

        if has_inferred_micro:
            # Microstructure was re-inferred in this run — write a new snapshot
            # only if values actually changed (avoids duplicates on re-runs)
            changed = latest is None or any(
                micro_vals.get(f) != latest.get(f) for f in _SNAP_FIELDS
            )
            if changed:
                snap_id = _db.save_microstructure_snapshot(
                    print_config_id=cfg_id,
                    mf=micro_vals.get("mf"),
                    ar=micro_vals.get("ar"),
                    a11=micro_vals.get("a11"),
                    a22=micro_vals.get("a22"),
                    a12=micro_vals.get("a12"),
                    a13=micro_vals.get("a13"),
                    a23=micro_vals.get("a23"),
                    provenance=micro_prov,
                    notes=notes,
                )
            else:
                snap_id = latest["id"]
        else:
            # Microstructure was fixed (loaded from a previous run) — reuse the
            # existing snapshot rather than writing a duplicate with "inputted" provenance
            snap_id = latest["id"] if latest else None

    # 3. Inference run
    run_id = _db.save_inference_run(
        print_config_id=cfg_id,
        stage=model_nm,
        inputs=all_in,
        outputs=outputs,
        solver_cfg=solver,
        loss=loss,
        microstructure_snap_id=snap_id,
        notes=notes,
    )

    # 4. Constituent properties
    def _latest_by_tag(ctype, cid):
        inferred: dict[str, float] = {}
        inputted: dict[str, float] = {}
        for p in _db.get_constituent_properties(ctype, cid, include_global=True):
            name = p["property_name"]
            tag  = p["source_tag"]
            if tag == "inferred" and name not in inferred:
                inferred[name] = float(p["value"])
            elif tag == "inputted" and name not in inputted:
                inputted[name] = float(p["value"])
        return inferred, inputted

    inf_f, inp_f = _latest_by_tag("fiber",   fiber_id)
    inf_p, inp_p = _latest_by_tag("polymer", polymer_id)

    # Free vars → inferred
    for field, value in free.items():
        if field in _FIBER_FIELDS:
            _db.save_constituent_property(
                constituent_type="fiber", constituent_id=fiber_id,
                property_name=field, value=value, source_tag="inferred",
                print_config_id=None, inference_run_id=run_id, notes=notes,
            )
        elif field in _POLYMER_FIELDS:
            _db.save_constituent_property(
                constituent_type="polymer", constituent_id=polymer_id,
                property_name=field, value=value, source_tag="inferred",
                print_config_id=None, inference_run_id=run_id, notes=notes,
            )

    # Fixed vars → inputted (skip if inferred already exists; skip if identical inputted exists)
    for field, value in fixed.items():
        fval = float(value)
        if field in _FIBER_FIELDS:
            if field in inf_f:
                continue
            if inp_f.get(field) == fval:
                continue
            _db.save_constituent_property(
                constituent_type="fiber", constituent_id=fiber_id,
                property_name=field, value=value, source_tag="inputted",
                print_config_id=None, inference_run_id=run_id, notes=notes,
            )
        elif field in _POLYMER_FIELDS:
            if field in inf_p:
                continue
            if inp_p.get(field) == fval:
                continue
            _db.save_constituent_property(
                constituent_type="polymer", constituent_id=polymer_id,
                property_name=field, value=value, source_tag="inputted",
                print_config_id=None, inference_run_id=run_id, notes=notes,
            )

    # 5. Composite properties
    # For thermoelastic: only save CTE outputs. Mechanical properties (E, G, nu)
    # are owned by the elastic model and must not be overwritten here.
    if model_nm == "thermoelastic":
        composite_outputs = {k: v for k, v in outputs.items() if k in _TE_CTE_OUTPUTS}
    else:
        composite_outputs = outputs

    for prop, value in composite_outputs.items():
        _db.save_composite_property(
            print_config_id=cfg_id,
            property_name=prop,
            value=value,
            source_tag="predicted",
            inference_run_id=run_id,
        )

    # 6. Experimental measurements (targets)
    for prop, value in targets.items():
        sigma = sigmas.get(prop, 0.0)
        _db.save_experimental_measurement(
            print_config_id=cfg_id,
            property_name=prop,
            value=value,
            uncertainty=float(sigma) if sigma else None,
            notes=notes,
        )

    return cfg_id


# ── save — forward prediction ──────────────────────────────────────────────────

def save_forward_result(
    model_name: str,
    inputs:     dict[str, float],
    outputs:    dict[str, float],
    card_id:    int,
    notes:      str = "",
) -> int:
    """Save a forward prediction result to an existing card. Returns inference_run_id."""
    run_id = _db.save_inference_run(
        print_config_id=card_id,
        stage=model_name,
        inputs=inputs,
        outputs=outputs,
        solver_cfg={},
        loss=0.0,
        notes=notes,
    )
    for prop, value in outputs.items():
        _db.save_composite_property(
            print_config_id=card_id,
            property_name=prop,
            value=value,
            source_tag="predicted",
            inference_run_id=run_id,
        )
    return run_id


# ── save — thermal inverse ─────────────────────────────────────────────────────

def save_thermal_result(
    result:     dict,
    fiber_id:   int,
    polymer_id: int,
    card_id:    Optional[int] = None,
    notes:      str = "",
) -> int:
    """Persist a thermal inverse result. Returns inference_run_id.
    Wraps db.save_thermal_inverse_results.

    result must contain:
        best_params  — ConstituentParams namedtuple with .p1, .p2, .l2, .t
        best_loss    — float
    card_id is the print_config_id to attach this result to (required).
    """
    if card_id is None:
        raise ValueError("save_thermal_result requires card_id (print_config_id)")

    params = result["best_params"]
    parametric_outputs = {
        "p1": float(params.p1),
        "p2": float(params.p2),
        "l2": float(params.l2),
        "t":  float(params.t),
    }

    return _db.save_thermal_inverse_results(
        print_config_id=card_id,
        fiber_id=fiber_id,
        polymer_id=polymer_id,
        parametric_outputs=parametric_outputs,
        solver_cfg={},
        loss=float(result["best_loss"]),
        notes=notes,
    )


# ── save — transfer ────────────────────────────────────────────────────────────

def save_transfer_result(
    result:            dict,
    source_card_id:    int,
    target_printer_id: int,
) -> int:
    """Create target print_config + save microstructure, inference run,
    experimental measurements, and all predicted composite properties.
    Returns new print_config_id.
    """
    source_card = _db.get_print_config(source_card_id)
    if source_card is None:
        raise ValueError(f"Source card id={source_card_id} not found")

    cfg_id = _db.get_or_create_print_config(
        name=source_card["name"],
        fiber_id=source_card["fiber_id"],
        polymer_id=source_card["polymer_id"],
        printer_id=target_printer_id,
    )

    micro = result.get("opt_microstructure", {})
    snap_id: Optional[int] = None
    if micro:
        snap_id = _db.save_microstructure_snapshot(
            print_config_id=cfg_id,
            mf=micro.get("fiber_massfrac") or micro.get("w_f"),
            ar=micro.get("ar") or micro.get("ar_f"),
            a11=micro.get("a11"),
            a22=micro.get("a22"),
            a12=micro.get("a12"),
            a13=micro.get("a13"),
            a23=micro.get("a23"),
            provenance={f: "inferred" for f in micro},
            notes=f"Transfer from card id={source_card_id}",
        )

    elastic_predicted = result.get("elastic_predicted", {})
    run_id = _db.save_inference_run(
        print_config_id=cfg_id,
        stage="transfer_elastic",
        inputs=result.get("constituent_inputs", {}),
        outputs=elastic_predicted,
        solver_cfg={"source_card_id": source_card_id},
        loss=result.get("elastic_error", 0.0),
        microstructure_snap_id=snap_id,
        notes=f"Property transfer from card id={source_card_id}",
    )

    # Save targets as experimental measurements
    for prop, value in result.get("target_outputs", {}).items():
        sigma = result.get("sigmas", {}).get(prop, 0.0)
        _db.save_experimental_measurement(
            print_config_id=cfg_id,
            property_name=prop,
            value=value,
            uncertainty=float(sigma) if sigma else None,
            notes="Transfer target",
        )

    # Save all forward predictions across all three models
    predictions = result.get("predictions", {})
    for model_name, preds in predictions.items():
        if preds is None:
            continue
        for prop, value in preds.items():
            _db.save_composite_property(
                print_config_id=cfg_id,
                property_name=prop,
                value=value,
                source_tag="predicted",
                inference_run_id=run_id,
            )

    return cfg_id
