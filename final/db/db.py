"""db.py — database helper module for DiffMicromechanics.

All reads/writes to data/micromechanics.db go through this module.

Units (stored in DB / JSON and passed directly to the surrogate):
    moduli       MPa
    density      kg/m³
    CTE          1/K      (entered and stored in SI units — no conversion needed)
    conductivity W/m·K

Model input units (what the surrogate expects):
    moduli       MPa
    density      kg/m³
    CTE          1/K      (no conversion needed)
    conductivity W/m·K    (no conversion needed)

── Reference table API ───────────────────────────────────────────────────────────
    get_all_fibers / get_fiber / fiber_model_inputs / add_fiber
    get_all_polymers / get_polymer / polymer_model_inputs / add_polymer
    get_all_printers / add_printer
    add_processing_condition / get_processing_condition / get_all_processing_conditions

── Material-card API ────────────────────────────────────────────────────────────
    create_print_config / get_print_config / get_all_print_configs
    save_microstructure_snapshot / get_latest_microstructure
        / get_all_microstructure_snapshots / get_best_microstructure
    save_inference_run / get_inference_runs
    save_constituent_property / get_constituent_properties
    save_experimental_measurement / get_experimental_measurements
    save_composite_property / get_composite_properties
        / get_current_composite_properties
    set_property_preference / get_canonical_value
    get_print_config_card

── Thermal helpers ───────────────────────────────────────────────────────────────
    save_thermal_inverse_results
    get_thermal_constituent_inputs
    import_thermal_csv
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

# db.py lives in final/db/ — go up one level to reach final/
HERE    = Path(__file__).parent.parent
DB_PATH = HERE / "data" / "micromechanics.db"


# ── connection ────────────────────────────────────────────────────────────────

@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def db_exists() -> bool:
    return DB_PATH.exists()


# ── fibers ────────────────────────────────────────────────────────────────────

def get_all_fibers() -> list[dict]:
    """Return all fibers as dicts (units: MPa, kg/m³)."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM fibers ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_fiber(fiber_id: int) -> Optional[dict]:
    """Return one fiber by id (units: MPa, kg/m³)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM fibers WHERE id = ?", (fiber_id,)
        ).fetchone()
    return dict(row) if row else None


def fiber_model_inputs(fiber_id: int) -> dict[str, float]:
    """
    Return the neat fiber properties in model input units (MPa, kg/m³, W/m·K).

    Model field mapping:
        e1           ← neat_E1  (MPa)
        e2           ← neat_E2  (MPa)
        g12          ← neat_G12 (MPa)
        f_nu12       ← neat_nu12
        f_nu23       ← neat_nu23
        fiber_density← neat_rho (kg/m³)
        rho_f        ← neat_rho (kg/m³)
        k_f1         ← neat_k1  (W/m·K)
        k_f2         ← neat_k2  (W/m·K)
    """
    f = get_fiber(fiber_id)
    if f is None:
        raise ValueError(f"Fiber id={fiber_id} not found")
    out = {
        "e1":            f["neat_E1"],
        "e2":            f["neat_E2"],
        "g12":           f["neat_G12"],
        "f_nu12":        f["neat_nu12"],
        "f_nu23":        f["neat_nu23"],
        "fiber_density": f["neat_rho"],
        "rho_f":         f["neat_rho"],
    }
    if f.get("neat_k1") is not None:
        out["k_f1"] = f["neat_k1"]
    if f.get("neat_k2") is not None:
        out["k_f2"] = f["neat_k2"]
    return out


def add_fiber(name: str, supplier: str, neat: dict) -> int:
    """Insert a new fiber. Returns new id."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO fibers
                (name, supplier, neat_E1, neat_E2, neat_G12,
                 neat_nu12, neat_nu23, neat_rho, neat_source, neat_notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                name, supplier,
                neat.get("E1"), neat.get("E2"), neat.get("G12"),
                neat.get("nu12"), neat.get("nu23"), neat.get("rho"),
                neat.get("source"), neat.get("notes"),
            ),
        )
    return cur.lastrowid


# ── polymers ──────────────────────────────────────────────────────────────────

def get_all_polymers() -> list[dict]:
    """Return all polymers as dicts (units: MPa, kg/m³)."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM polymers ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_polymer(polymer_id: int) -> Optional[dict]:
    """Return one polymer by id (units: MPa, kg/m³)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM polymers WHERE id = ?", (polymer_id,)
        ).fetchone()
    return dict(row) if row else None


def polymer_model_inputs(polymer_id: int) -> dict[str, float]:
    """
    Return neat polymer properties in model input units (MPa, kg/m³, W/m·K).

    Model field mapping:
        matrix_modulus  ← neat_E1  (MPa)
        matrix_poisson  ← neat_nu12
        matrix_density  ← neat_rho (kg/m³)
        rho_m           ← neat_rho (kg/m³)
        k_m             ← neat_k   (W/m·K)
    """
    p = get_polymer(polymer_id)
    if p is None:
        raise ValueError(f"Polymer id={polymer_id} not found")
    out = {
        "matrix_modulus":  p["neat_E1"],
        "matrix_poisson":  p["neat_nu12"],
        "matrix_density":  p["neat_rho"],
        "rho_m":           p["neat_rho"],
    }
    if p.get("neat_k") is not None:
        out["k_m"] = p["neat_k"]
    return out


def add_polymer(name: str, supplier: str, neat: dict) -> int:
    """Insert a new polymer. Returns new id."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO polymers
                (name, supplier, neat_E1, neat_E2, neat_G12,
                 neat_nu12, neat_rho, neat_source, neat_notes)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                name, supplier,
                neat.get("E1"), neat.get("E2"), neat.get("G12"),
                neat.get("nu12"), neat.get("rho"),
                neat.get("source"), neat.get("notes"),
            ),
        )
    return cur.lastrowid


# ── printers ──────────────────────────────────────────────────────────────────

def get_all_printers() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM printers ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def add_printer(name: str, manufacturer: str = "", notes: str = "") -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO printers (name, manufacturer, notes) VALUES (?,?,?)",
            (name, manufacturer, notes),
        )
    return cur.lastrowid


# ── processing conditions ─────────────────────────────────────────────────────

def add_processing_condition(
    bead_width:      Optional[float] = None,
    bead_height:     Optional[float] = None,
    nozzle_diameter: Optional[float] = None,
    speed:           Optional[float] = None,
    notes:           str = "",
) -> int:
    """Insert a processing condition record. Returns new id."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO processing_conditions
                (bead_width, bead_height, nozzle_diameter, speed, notes, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (bead_width, bead_height, nozzle_diameter, speed, notes,
             datetime.now().isoformat()),
        )
    return cur.lastrowid


def get_processing_condition(pc_id: int) -> Optional[dict]:
    """Return one processing condition by id."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM processing_conditions WHERE id = ?", (pc_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_processing_conditions() -> list[dict]:
    """Return all processing conditions, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM processing_conditions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ── print configs (material cards) ────────────────────────────────────────────

def create_print_config(
    name:                    str,
    fiber_id:                int,
    polymer_id:              int,
    printer_id:              Optional[int] = None,
    processing_condition_id: Optional[int] = None,
    notes:                   str = "",
) -> int:
    """Create a new print config (material card). Returns new id."""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO print_configs
                (name, fiber_id, polymer_id, printer_id,
                 processing_condition_id, notes, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (name, fiber_id, polymer_id, printer_id,
             processing_condition_id, notes,
             datetime.now().isoformat()),
        )
    return cur.lastrowid


def get_print_config(print_config_id: int) -> Optional[dict]:
    """Return one print config by id."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM print_configs WHERE id = ?", (print_config_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_print_configs() -> list[dict]:
    """Return all print configs, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM print_configs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_print_configs_for_printer(printer_id: int) -> list[dict]:
    """Return all print configs for a given printer, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM print_configs WHERE printer_id = ? ORDER BY created_at DESC",
            (printer_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_or_create_print_config(
    name: str, fiber_id: int, polymer_id: int, printer_id: int,
) -> int:
    """Return existing print_config id for (name, fiber, polymer, printer) or create it."""
    with _connect() as conn:
        row = conn.execute(
            """SELECT id FROM print_configs
               WHERE name = ? AND fiber_id = ? AND polymer_id = ? AND printer_id = ?
               LIMIT 1""",
            (name, fiber_id, polymer_id, printer_id),
        ).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            """INSERT INTO print_configs
               (name, fiber_id, polymer_id, printer_id, notes, created_at)
               VALUES (?,?,?,?,?,?)""",
            (name, fiber_id, polymer_id, printer_id,
             "Created by property transfer", datetime.now().isoformat()),
        )
        return cur.lastrowid


# ── microstructure snapshots ───────────────────────────────────────────────────

def save_microstructure_snapshot(
    print_config_id:  int,
    mf:               Optional[float],
    ar:               Optional[float],
    a11:              Optional[float],
    a22:              Optional[float],
    a12:              Optional[float],
    a13:              Optional[float],
    a23:              Optional[float],
    provenance:       Optional[dict] = None,
    inference_run_id: Optional[int] = None,
    notes:            str = "",
) -> int:
    """
    Insert a microstructure snapshot for a print config.

    `provenance` is a dict with per-field source tags, e.g.:
        {"mf": "inputted", "ar": "inputted", "a11": "inferred", ...}
    Any missing fields default to None in the stored JSON.

    Returns the new snapshot id.
    """
    _FIELDS = ("mf", "ar", "a11", "a22", "a12", "a13", "a23")
    prov = {f: (provenance or {}).get(f) for f in _FIELDS}

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO microstructure_snapshots
                (print_config_id, mf, ar, a11, a22, a12, a13, a23,
                 provenance_json, inference_run_id, notes, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (print_config_id, mf, ar, a11, a22, a12, a13, a23,
             json.dumps(prov), inference_run_id, notes,
             datetime.now().isoformat()),
        )
    return cur.lastrowid


def get_latest_microstructure(print_config_id: int) -> Optional[dict]:
    """Return the most recently created microstructure snapshot for a print config."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM microstructure_snapshots
            WHERE print_config_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (print_config_id,),
        ).fetchone()
    if row is None:
        return None
    r = dict(row)
    if r.get("provenance_json"):
        r["provenance"] = json.loads(r["provenance_json"])
    return r


def get_all_microstructure_snapshots(print_config_id: int) -> list[dict]:
    """Return all microstructure snapshots for a print config, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM microstructure_snapshots
            WHERE print_config_id = ?
            ORDER BY created_at DESC
            """,
            (print_config_id,),
        ).fetchall()
    result = []
    for row in rows:
        r = dict(row)
        if r.get("provenance_json"):
            r["provenance"] = json.loads(r["provenance_json"])
        result.append(r)
    return result


def get_best_microstructure(print_config_id: int) -> Optional[dict]:
    """Return the microstructure snapshot linked to the lowest-loss inference run.
    Falls back to the most recent snapshot if no loss-linked snapshot exists."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT ms.* FROM microstructure_snapshots ms
            JOIN inference_runs ir ON ir.microstructure_snap_id = ms.id
            WHERE ms.print_config_id = ? AND ir.loss IS NOT NULL
            ORDER BY ir.loss ASC
            LIMIT 1
            """,
            (print_config_id,),
        ).fetchone()
    if row is None:
        return get_latest_microstructure(print_config_id)
    r = dict(row)
    if r.get("provenance_json"):
        r["provenance"] = json.loads(r["provenance_json"])
    return r


# ── inference runs ─────────────────────────────────────────────────────────────

def save_inference_run(
    print_config_id:       int,
    stage:                 str,
    inputs:                dict,
    outputs:               dict,
    solver_cfg:            dict,
    loss:                  float,
    microstructure_snap_id: Optional[int] = None,
    notes:                 str = "",
) -> int:
    """
    Record one inference run.

    `stage` values: 'elastic' | 'thermoelastic' | 'thermal_inverse' | 'thermal_forward'

    Returns the new inference_run id.
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO inference_runs
                (print_config_id, stage, microstructure_snap_id,
                 inputs_json, outputs_json, solver_json,
                 loss, notes, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                print_config_id,
                stage,
                microstructure_snap_id,
                json.dumps(inputs),
                json.dumps(outputs),
                json.dumps(solver_cfg),
                loss,
                notes,
                datetime.now().isoformat(),
            ),
        )
    return cur.lastrowid


def get_inference_runs(
    print_config_id: int,
    stage:           Optional[str] = None,
) -> list[dict]:
    """Return all inference runs for a print config, newest first. Filter by stage if given."""
    with _connect() as conn:
        if stage:
            rows = conn.execute(
                """
                SELECT * FROM inference_runs
                WHERE print_config_id = ? AND stage = ?
                ORDER BY created_at DESC
                """,
                (print_config_id, stage),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM inference_runs
                WHERE print_config_id = ?
                ORDER BY created_at DESC
                """,
                (print_config_id,),
            ).fetchall()
    result = []
    for row in rows:
        r = dict(row)
        for key in ("inputs_json", "outputs_json", "solver_json"):
            if r.get(key):
                r[key.replace("_json", "")] = json.loads(r[key])
        result.append(r)
    return result


# ── constituent property values ────────────────────────────────────────────────

def save_constituent_property(
    constituent_type: str,
    constituent_id:   int,
    property_name:    str,
    value:            float,
    source_tag:       str,
    unit:             str = "",
    print_config_id:  Optional[int] = None,
    inference_run_id: Optional[int] = None,
    notes:            str = "",
) -> int:
    """
    Store one constituent property value with provenance.

    `constituent_type`: 'fiber' | 'polymer'
    `source_tag`:       'web' | 'inputted' | 'inferred' | 'predicted'
    `print_config_id`:  None = global (printer-agnostic)

    Returns the new row id.
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO constituent_property_values
                (constituent_type, constituent_id, print_config_id,
                 property_name, value, unit, source_tag,
                 inference_run_id, notes, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                constituent_type, constituent_id, print_config_id,
                property_name, value, unit, source_tag,
                inference_run_id, notes,
                datetime.now().isoformat(),
            ),
        )
    return cur.lastrowid


def get_constituent_properties(
    constituent_type: str,
    constituent_id:   int,
    property_name:    Optional[str] = None,
    print_config_id:  Optional[int] = None,
    include_global:   bool = True,
) -> list[dict]:
    """
    Return constituent property values, newest first.

    If `print_config_id` is given and `include_global` is True, returns rows
    for that card AND global rows (print_config_id IS NULL), sorted newest first.
    """
    with _connect() as conn:
        if print_config_id is not None and include_global:
            extra = "AND (print_config_id = ? OR print_config_id IS NULL)"
            params: tuple = (constituent_type, constituent_id, print_config_id)
        elif print_config_id is not None:
            extra = "AND print_config_id = ?"
            params = (constituent_type, constituent_id, print_config_id)
        else:
            extra = "AND print_config_id IS NULL"
            params = (constituent_type, constituent_id)

        prop_filter = "AND property_name = ?" if property_name else ""
        if property_name:
            params = params + (property_name,)

        rows = conn.execute(
            f"""
            SELECT * FROM constituent_property_values
            WHERE constituent_type = ? AND constituent_id = ?
              {extra}
              {prop_filter}
            ORDER BY created_at DESC
            """,
            params,
        ).fetchall()
    return [dict(r) for r in rows]


# ── experimental measurements ──────────────────────────────────────────────────

def save_experimental_measurement(
    print_config_id: int,
    property_name:   str,
    value:           float,
    unit:            str = "",
    uncertainty:     Optional[float] = None,
    temperature_C:   Optional[float] = None,
    conditions:      Optional[dict] = None,
    reference:       str = "",
    date:            Optional[str] = None,
    notes:           str = "",
) -> int:
    """
    Record one experimental measurement for a print config.

    Returns the new measurement id.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO experimental_measurements
                (print_config_id, property_name, value, unit,
                 uncertainty, temperature_C, conditions_json,
                 reference, date, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                print_config_id, property_name, value, unit,
                uncertainty, temperature_C,
                json.dumps(conditions) if conditions else None,
                reference, date, notes,
            ),
        )
    return cur.lastrowid


def get_experimental_measurements(
    print_config_id: int,
    property_name:   Optional[str] = None,
) -> list[dict]:
    """Return experimental measurements for a print config, newest first."""
    with _connect() as conn:
        if property_name:
            rows = conn.execute(
                """
                SELECT * FROM experimental_measurements
                WHERE print_config_id = ? AND property_name = ?
                ORDER BY date DESC
                """,
                (print_config_id, property_name),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM experimental_measurements
                WHERE print_config_id = ?
                ORDER BY date DESC
                """,
                (print_config_id,),
            ).fetchall()
    result = []
    for row in rows:
        r = dict(row)
        if r.get("conditions_json"):
            r["conditions"] = json.loads(r["conditions_json"])
        result.append(r)
    return result


# ── composite property values ──────────────────────────────────────────────────

def save_composite_property(
    print_config_id:  int,
    property_name:    str,
    value:            float,
    source_tag:       str,
    unit:             str = "",
    inference_run_id: Optional[int] = None,
    measurement_id:   Optional[int] = None,
    temperature_C:    Optional[float] = None,
) -> int:
    """
    Store a composite property value.

    `source_tag`: 'predicted' | 'experimental' | 'inputted'

    Returns new row id.
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO composite_property_values
                (print_config_id, property_name, value, unit,
                 source_tag, inference_run_id, measurement_id,
                 temperature_C, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                print_config_id, property_name, value, unit,
                source_tag, inference_run_id, measurement_id,
                temperature_C,
                datetime.now().isoformat(),
            ),
        )
        row_id = cur.lastrowid
        conn.execute(
            """
            INSERT INTO current_composite_properties
                (print_config_id, property_name, value, unit,
                 source_tag, source_run_id, updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT (print_config_id, property_name)
            DO UPDATE SET
                value         = excluded.value,
                unit          = excluded.unit,
                source_tag    = excluded.source_tag,
                source_run_id = excluded.source_run_id,
                updated_at    = excluded.updated_at
            """,
            (print_config_id, property_name, value, unit,
             source_tag, inference_run_id,
             datetime.now().isoformat()),
        )
    return row_id


def get_composite_properties(
    print_config_id: int,
    property_name:   Optional[str] = None,
    source_tag:      Optional[str] = None,
) -> list[dict]:
    """Return composite property values for a print config, newest first."""
    filters = ["print_config_id = ?"]
    params: list = [print_config_id]
    if property_name:
        filters.append("property_name = ?")
        params.append(property_name)
    if source_tag:
        filters.append("source_tag = ?")
        params.append(source_tag)

    where = " AND ".join(filters)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM composite_property_values WHERE {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_current_composite_properties(print_config_id: int) -> list[dict]:
    """Return one row per property — the current (most-recently-saved) value.
    Much faster than get_composite_properties() for large datasets."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM current_composite_properties WHERE print_config_id = ?",
            (print_config_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── property preferences ───────────────────────────────────────────────────────

def set_property_preference(
    print_config_id:  int,
    property_name:    str,
    preferred_source: str,
) -> None:
    """
    Set which source to prefer for a given property on a print config.

    `preferred_source`: 'experimental' | 'predicted' | 'inferred' | 'web' | 'inputted'
    """
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO property_preferences (print_config_id, property_name, preferred_source)
            VALUES (?,?,?)
            ON CONFLICT (print_config_id, property_name)
            DO UPDATE SET preferred_source = excluded.preferred_source
            """,
            (print_config_id, property_name, preferred_source),
        )


def get_canonical_value(
    print_config_id: int,
    property_name:   str,
) -> Optional[dict]:
    """
    Return the canonical value for a composite property, respecting the
    resolution hierarchy:

      1. property_preferences override (highest priority)
      2. experimental (most recent measurement)
      3. predicted (most recent surrogate output)
      4. inputted

    Returns a dict with 'value', 'source_tag', and 'created_at' (or 'date'),
    or None if no value is found.
    """
    with _connect() as conn:
        pref_row = conn.execute(
            """
            SELECT preferred_source FROM property_preferences
            WHERE print_config_id = ? AND property_name = ?
            """,
            (print_config_id, property_name),
        ).fetchone()
        preferred_source = pref_row["preferred_source"] if pref_row else None

        if preferred_source:
            if preferred_source == "experimental":
                row = conn.execute(
                    """
                    SELECT value, 'experimental' AS source_tag, date AS created_at
                    FROM experimental_measurements
                    WHERE print_config_id = ? AND property_name = ?
                    ORDER BY date DESC LIMIT 1
                    """,
                    (print_config_id, property_name),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT value, source_tag, created_at
                    FROM composite_property_values
                    WHERE print_config_id = ? AND property_name = ? AND source_tag = ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (print_config_id, property_name, preferred_source),
                ).fetchone()
            if row:
                return dict(row)

        # Default hierarchy: experimental → predicted → inputted
        exp_row = conn.execute(
            """
            SELECT value, 'experimental' AS source_tag, date AS created_at
            FROM experimental_measurements
            WHERE print_config_id = ? AND property_name = ?
            ORDER BY date DESC LIMIT 1
            """,
            (print_config_id, property_name),
        ).fetchone()
        if exp_row:
            return dict(exp_row)

        for src in ("predicted", "inputted"):
            comp_row = conn.execute(
                """
                SELECT value, source_tag, created_at
                FROM composite_property_values
                WHERE print_config_id = ? AND property_name = ? AND source_tag = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (print_config_id, property_name, src),
            ).fetchone()
            if comp_row:
                return dict(comp_row)

    return None


# ── print config card view ─────────────────────────────────────────────────────

def get_print_config_card(print_config_id: int) -> dict:
    """
    Return a comprehensive material card view for a print config.

    Returns:
        {
            "config":                 dict (print_configs row),
            "fiber":                  dict (fibers row),
            "polymer":                dict (polymers row),
            "printer":                dict | None (printers row),
            "processing_condition":   dict | None (processing_conditions row),
            "microstructure":         dict | None (latest snapshot),
            "inference_runs":  list[dict],
            "constituent_properties": {
                "fiber": list[dict],
                "polymer": list[dict],
            },
            "experimental_measurements": list[dict],
            "composite_properties":      list[dict],
        }
    """
    cfg = get_print_config(print_config_id)
    if cfg is None:
        raise ValueError(f"print_config id={print_config_id} not found")

    fiber   = get_fiber(cfg["fiber_id"])
    polymer = get_polymer(cfg["polymer_id"])

    printer = None
    if cfg.get("printer_id"):
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM printers WHERE id = ?", (cfg["printer_id"],)
            ).fetchone()
        printer = dict(row) if row else None

    processing_condition = None
    if cfg.get("processing_condition_id"):
        processing_condition = get_processing_condition(cfg["processing_condition_id"])

    return {
        "config":                cfg,
        "fiber":                 fiber,
        "polymer":               polymer,
        "printer":               printer,
        "processing_condition":  processing_condition,
        "microstructure": get_latest_microstructure(print_config_id),
        "inference_runs": get_inference_runs(print_config_id),
        "constituent_properties": {
            "fiber":   get_constituent_properties("fiber",   cfg["fiber_id"],   print_config_id=print_config_id),
            "polymer": get_constituent_properties("polymer", cfg["polymer_id"], print_config_id=print_config_id),
        },
        "experimental_measurements": get_experimental_measurements(print_config_id),
        "composite_properties":      get_composite_properties(print_config_id),
    }


# ── thermal inverse helpers ────────────────────────────────────────────────────

def save_thermal_k_predictions(
    print_config_id:  int,
    inference_run_id: int,
    temperatures,
    K_pred,
) -> int:
    """
    Store one K-vs-T prediction curve in thermal_k_predictions.

    Parameters
    ----------
    temperatures : array-like of float, length n  (°C)
    K_pred       : array-like shape (n, 3) — columns K11, K22, K33 (W/m·K)
                   OR dict {"K11": [...], "K22": [...], "K33": [...]}

    Returns new row id.
    """
    import json as _json
    import numpy as _np

    T = _np.asarray(temperatures).tolist()
    if isinstance(K_pred, dict):
        K11 = list(K_pred["K11"])
        K22 = list(K_pred["K22"])
        K33 = list(K_pred["K33"])
    else:
        K = _np.asarray(K_pred)
        K11, K22, K33 = K[:, 0].tolist(), K[:, 1].tolist(), K[:, 2].tolist()

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO thermal_k_predictions
                (print_config_id, inference_run_id,
                 temperatures_json, K11_json, K22_json, K33_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                print_config_id,
                inference_run_id,
                _json.dumps(T),
                _json.dumps(K11),
                _json.dumps(K22),
                _json.dumps(K33),
                datetime.now().isoformat(),
            ),
        )
    return cur.lastrowid


def get_thermal_k_predictions(print_config_id: int) -> list[dict]:
    """
    Return all K-vs-T prediction curves for a card, newest first.

    Each entry has keys:
        id, print_config_id, inference_run_id, created_at
        temperatures  : list[float]  (°C)
        K11           : list[float]  (W/m·K)
        K22           : list[float]
        K33           : list[float]
    """
    import json as _json

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, print_config_id, inference_run_id,
                   temperatures_json, K11_json, K22_json, K33_json, created_at
            FROM thermal_k_predictions
            WHERE print_config_id = ?
            ORDER BY created_at DESC
            """,
            (print_config_id,),
        ).fetchall()

    result = []
    for r in rows:
        result.append({
            "id":              r["id"],
            "print_config_id": r["print_config_id"],
            "inference_run_id": r["inference_run_id"],
            "created_at":      r["created_at"],
            "temperatures":    _json.loads(r["temperatures_json"]),
            "K11":             _json.loads(r["K11_json"]),
            "K22":             _json.loads(r["K22_json"]),
            "K33":             _json.loads(r["K33_json"]),
        })
    return result


def save_thermal_inverse_results(
    print_config_id:       int,
    fiber_id:              int,
    polymer_id:            int,
    parametric_outputs:    dict,
    solver_cfg:            dict,
    loss:                  float,
    microstructure_snap_id: Optional[int] = None,
    notes:                 str = "",
    temperatures=None,
    K_pred=None,
) -> int:
    """
    Save the results of a thermal inverse run.

    `parametric_outputs` should contain any subset of:
        k_f1, k_f2  — fiber conductivities (W/m·K)
        k_m         — matrix conductivity (W/m·K)
        p1, p2, l2, t — parametric model coefficients

    `temperatures` and `K_pred` (shape n×3, columns K11/K22/K33) are stored
    in thermal_k_predictions (one row per save, cross-referenced by
    print_config_id and inference_run_id).

    Stores:
      - one inference_run   (stage='thermal_inverse')
      - one thermal_k_predictions row (when temperatures + K_pred provided)
      - one constituent_property_value per inferred scalar property

    Returns the inference_run id.
    """
    run_id = save_inference_run(
        print_config_id=print_config_id,
        stage="thermal_inverse",
        inputs={},
        outputs=dict(parametric_outputs),   # scalar outputs only — no curve blobs
        solver_cfg=solver_cfg,
        loss=loss,
        microstructure_snap_id=microstructure_snap_id,
        notes=notes,
    )

    # K vs T prediction curve — one row in thermal_k_predictions
    if temperatures is not None and K_pred is not None:
        save_thermal_k_predictions(
            print_config_id=print_config_id,
            inference_run_id=run_id,
            temperatures=temperatures,
            K_pred=K_pred,
        )

    # Fiber conductivities — derived from parametric model, global
    # l2 = fiber longitudinal conductivity (k_f1)
    # l2/t = fiber transverse conductivity (k_f2)
    l2 = parametric_outputs.get("l2")
    t  = parametric_outputs.get("t")
    if l2 is not None:
        save_constituent_property(
            constituent_type="fiber",
            constituent_id=fiber_id,
            property_name="k_f1",
            value=l2,
            source_tag="inferred",
            unit="W/m·K",
            print_config_id=None,
            inference_run_id=run_id,
            notes=notes,
        )
        if t is not None and t > 0:
            save_constituent_property(
                constituent_type="fiber",
                constituent_id=fiber_id,
                property_name="k_f2",
                value=l2 / t,
                source_tag="inferred",
                unit="W/m·K",
                print_config_id=None,
                inference_run_id=run_id,
                notes=notes,
            )

    # Polymer conductivity model coefficients — global
    # K_m(T) = p1 * sqrt(T / T_ref) + p2
    for prop in ("p1", "p2"):
        if prop in parametric_outputs:
            save_constituent_property(
                constituent_type="polymer",
                constituent_id=polymer_id,
                property_name=prop,
                value=parametric_outputs[prop],
                source_tag="inferred",
                unit="W/m·K",
                print_config_id=None,
                inference_run_id=run_id,
                notes=notes,
            )

    # Matrix conductivity at room temperature — global.
    # If k_m was not explicitly provided, compute from p1/p2 parametric model:
    #   K_m(T) = p1 * sqrt(T / T_ref) + p2,  T_ref = 1.0 °C (Thomas et al. 2024)
    # Evaluated at T = 25 °C (room temperature).
    p1 = parametric_outputs.get("p1")
    p2 = parametric_outputs.get("p2")
    k_m = parametric_outputs.get("k_m")
    if k_m is None and p1 is not None and p2 is not None:
        import math as _math
        T_room_degC = 25.0
        T_ref       = 1.0
        k_m = p1 * _math.sqrt(T_room_degC / T_ref) + p2
    if k_m is not None:
        save_constituent_property(
            constituent_type="polymer",
            constituent_id=polymer_id,
            property_name="k_m",
            value=k_m,
            source_tag="inferred",
            unit="W/m·K",
            print_config_id=None,
            inference_run_id=run_id,
            notes=notes,
        )

    return run_id


def get_thermal_constituent_inputs(
    print_config_id: int,
    fiber_id:        int,
    polymer_id:      int,
) -> dict[str, Optional[float]]:
    """
    Retrieve constituent thermal conductivity inputs ready for the forward surrogate.

    Resolution order for k_f1, k_f2, k_m:
      1. Inferred values in constituent_property_values (most recent)
      2. Neat values in fibers/polymers tables (web/seed data)

    Returns a dict with keys: k_f1, k_f2, k_m (values may be None if not found).
    """
    result: dict[str, Optional[float]] = {"k_f1": None, "k_f2": None, "k_m": None}

    # Fiber k_f1, k_f2
    for prop in ("k_f1", "k_f2"):
        rows = get_constituent_properties(
            "fiber", fiber_id, property_name=prop,
            print_config_id=print_config_id, include_global=True,
        )
        inferred = [r for r in rows if r["source_tag"] == "inferred"]
        if inferred:
            result[prop] = inferred[0]["value"]
        else:
            fiber = get_fiber(fiber_id)
            if fiber:
                result[prop] = fiber.get(f"neat_{prop}")  # neat_k1 / neat_k2

    # Matrix k_m
    rows = get_constituent_properties(
        "polymer", polymer_id, property_name="k_m",
        print_config_id=print_config_id, include_global=True,
    )
    inferred = [r for r in rows if r["source_tag"] == "inferred"]
    if inferred:
        result["k_m"] = inferred[0]["value"]
    else:
        polymer = get_polymer(polymer_id)
        if polymer:
            result["k_m"] = polymer.get("neat_k")

    return result


def import_thermal_csv(
    print_config_id:  int,
    csv_path:         str,
    temperature_col:  str = "T_C",
    property_map:     Optional[dict] = None,
    uncertainty_col:  Optional[str] = None,
    reference:        str = "",
    notes:            str = "",
) -> int:
    """
    Import thermal conductivity measurements from a CSV file.

    `property_map` maps CSV column names → property_name stored in DB.
    Default: {"K11": "K11", "K22": "K22", "K33": "K33"}

    Each row in the CSV produces one experimental_measurement row per property column.

    Returns the number of rows imported.
    """
    import csv

    if property_map is None:
        property_map = {"K11": "K11", "K22": "K22", "K33": "K33"}

    count = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            temp = float(row[temperature_col]) if temperature_col in row else None
            unc  = float(row[uncertainty_col]) if (uncertainty_col and uncertainty_col in row) else None
            for csv_col, prop_name in property_map.items():
                if csv_col not in row or row[csv_col].strip() == "":
                    continue
                save_experimental_measurement(
                    print_config_id=print_config_id,
                    property_name=prop_name,
                    value=float(row[csv_col]),
                    unit="W/m·K",
                    uncertainty=unc,
                    temperature_C=temp,
                    reference=reference,
                    notes=notes,
                )
                count += 1
    return count
