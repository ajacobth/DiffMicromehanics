"""db.py — database helper module for DiffMicromechanics.

All reads/writes to data/micromechanics.db go through this module.

Units (stored in DB / JSON and passed directly to the surrogate):
    moduli    MPa
    density   kg/m³
    CTE       µ/K
    conductivity  W/m·K

Model input units (what the surrogate expects):
    moduli    MPa
    density   kg/m³
    CTE       1/K      (× 1e-6 from µ/K)
    conductivity  W/m·K  (no conversion needed)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

HERE    = Path(__file__).parent
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
    Return the neat fiber properties in model input units (MPa, kg/m³).

    Model field mapping:
        e1           ← neat_E1  (MPa)
        e2           ← neat_E2  (MPa)
        g12          ← neat_G12 (MPa)
        f_nu12       ← neat_nu12
        f_nu23       ← neat_nu23
        fiber_density← neat_rho (kg/m³)
        rho_f        ← neat_rho (kg/m³)
    """
    f = get_fiber(fiber_id)
    if f is None:
        raise ValueError(f"Fiber id={fiber_id} not found")
    return {
        "e1":            f["neat_E1"],
        "e2":            f["neat_E2"],
        "g12":           f["neat_G12"],
        "f_nu12":        f["neat_nu12"],
        "f_nu23":        f["neat_nu23"],
        "fiber_density": f["neat_rho"],
        "rho_f":         f["neat_rho"],
    }


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
    Return neat polymer properties in model input units (MPa, kg/m³).

    Model field mapping:
        matrix_modulus  ← neat_E1  (MPa)
        matrix_poisson  ← neat_nu12
        matrix_density  ← neat_rho (kg/m³)
        rho_m           ← neat_rho (kg/m³)
    """
    p = get_polymer(polymer_id)
    if p is None:
        raise ValueError(f"Polymer id={polymer_id} not found")
    return {
        "matrix_modulus":  p["neat_E1"],
        "matrix_poisson":  p["neat_nu12"],
        "matrix_density":  p["neat_rho"],
        "rho_m":           p["neat_rho"],
    }


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


# ── microstructure ────────────────────────────────────────────────────────────

def get_or_create_microstructure(
    fiber_id:    int,
    polymer_id:  int,
    Vf:          float,
    ar:          float,
    a11:         float,
    a22:         float,
    a12:         float,
    a13:         float,
    a23:         float,
    printer_id:  Optional[int] = None,
    w_f:         Optional[float] = None,
    orientation_source: str = "",
    notes:       str = "",
) -> int:
    """
    Return id of an existing matching microstructure or insert a new one.
    Matching is done on fiber_id, polymer_id, Vf, ar, and orientation tensor.
    """
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM microstructure
            WHERE fiber_id=? AND polymer_id=?
              AND abs(Vf  - ?) < 1e-6
              AND abs(ar  - ?) < 1e-4
              AND abs(a11 - ?) < 1e-6
              AND abs(a22 - ?) < 1e-6
              AND abs(a12 - ?) < 1e-6
              AND abs(a13 - ?) < 1e-6
              AND abs(a23 - ?) < 1e-6
            LIMIT 1
            """,
            (fiber_id, polymer_id, Vf, ar, a11, a22, a12, a13, a23),
        ).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            """
            INSERT INTO microstructure
                (fiber_id, polymer_id, printer_id, Vf, w_f, ar,
                 a11, a22, a12, a13, a23, orientation_source, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (fiber_id, polymer_id, printer_id, Vf, w_f, ar,
             a11, a22, a12, a13, a23, orientation_source, notes),
        )
        return cur.lastrowid


# ── experiments & inferred properties ────────────────────────────────────────

def save_experiment(
    microstructure_id: int,
    property_type:     str,
    inferred:          dict,
    solver_cfg:        dict,
    loss:              float,
    model:             str,
    tags:              Optional[dict] = None,
    notes:             str = "",
    date:              Optional[str] = None,
) -> int:
    """
    Insert one experiment + its inferred_properties row.

    `inferred` keys (all optional):
        E1, E2, G12, nu12  — MPa / dimensionless
        CTE1, CTE2         — µ/K
        TC1, TC2           — W/m·K

    Returns the new experiment id.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    with _connect() as conn:
        exp_cur = conn.execute(
            """
            INSERT INTO experiments
                (microstructure_id, property_type, date, notes, tags_json)
            VALUES (?,?,?,?,?)
            """,
            (
                microstructure_id,
                property_type,
                date,
                notes,
                json.dumps(tags) if tags else None,
            ),
        )
        exp_id = exp_cur.lastrowid

        conn.execute(
            """
            INSERT INTO inferred_properties
                (experiment_id,
                 E1, E2, G12, nu12,
                 CTE1, CTE2,
                 TC1, TC2,
                 loss, model, solver_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                exp_id,
                inferred.get("E1"),
                inferred.get("E2"),
                inferred.get("G12"),
                inferred.get("nu12"),
                inferred.get("CTE1"),
                inferred.get("CTE2"),
                inferred.get("TC1"),
                inferred.get("TC2"),
                loss,
                model,
                json.dumps(solver_cfg),
            ),
        )

    return exp_id


# ── material card queries ─────────────────────────────────────────────────────

def get_material_card(fiber_id: int, polymer_id: int) -> dict:
    """
    Return a dict with:
      - fiber: neat properties
      - polymer: neat properties
      - best_inferred: lowest-loss inferred_properties row for this pair (or None)
      - experiments: list of all experiments for this pair, newest first
    """
    fiber   = get_fiber(fiber_id)
    polymer = get_polymer(polymer_id)

    with _connect() as conn:
        exps = conn.execute(
            """
            SELECT e.id, e.property_type, e.date, e.notes, e.tags_json,
                   ip.E1, ip.E2, ip.G12, ip.nu12,
                   ip.CTE1, ip.CTE2, ip.TC1, ip.TC2,
                   ip.loss, ip.model
            FROM experiments e
            JOIN inferred_properties ip ON ip.experiment_id = e.id
            JOIN microstructure m ON m.id = e.microstructure_id
            WHERE m.fiber_id = ? AND m.polymer_id = ?
            ORDER BY e.date DESC, ip.loss ASC
            """,
            (fiber_id, polymer_id),
        ).fetchall()

        best = conn.execute(
            """
            SELECT ip.*
            FROM inferred_properties ip
            JOIN experiments e ON e.id = ip.experiment_id
            JOIN microstructure m ON m.id = e.microstructure_id
            WHERE m.fiber_id = ? AND m.polymer_id = ?
            ORDER BY ip.loss ASC
            LIMIT 1
            """,
            (fiber_id, polymer_id),
        ).fetchone()

    return {
        "fiber":         fiber,
        "polymer":       polymer,
        "best_inferred": dict(best) if best else None,
        "experiments":   [dict(e) for e in exps],
    }


def get_experiments_for_pair(fiber_id: int, polymer_id: int) -> list[dict]:
    """All experiments for a fiber/polymer pair, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.property_type, e.date, e.notes, e.tags_json,
                   ip.loss, ip.model,
                   ip.E1, ip.E2, ip.G12, ip.nu12,
                   ip.CTE1, ip.CTE2, ip.TC1, ip.TC2
            FROM experiments e
            JOIN inferred_properties ip ON ip.experiment_id = e.id
            JOIN microstructure m ON m.id = e.microstructure_id
            WHERE m.fiber_id = ? AND m.polymer_id = ?
            ORDER BY e.date DESC, ip.loss ASC
            """,
            (fiber_id, polymer_id),
        ).fetchall()
    return [dict(r) for r in rows]


def _um_per_k_to_per_k(v: Optional[float]) -> Optional[float]:
    """µ/K → 1/K"""
    return v * 1e-6 if v is not None else None
