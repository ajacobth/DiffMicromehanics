"""init_db.py — create and seed data/micromechanics.db from the JSON seed files.

Run once (or re-run to reset):
    python init_db.py

Safe to re-run: drops and recreates all tables, then re-seeds from JSON.
No external dependencies — uses only Python stdlib (sqlite3, json, pathlib).

Tables
------
Reference (3):   fibers, polymers, printers
Material card (8): print_configs, microstructure_snapshots, inference_runs,
                   constituent_property_values, composite_property_values,
                   experimental_measurements, property_preferences,
                   current_composite_properties
"""

import json
import sqlite3
from pathlib import Path

HERE     = Path(__file__).parent
DATA_DIR = HERE / "data"
DB_PATH  = DATA_DIR / "micromechanics.db"


# ── schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS fibers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    supplier        TEXT,
    neat_E1         REAL,           -- MPa
    neat_E2         REAL,           -- MPa
    neat_G12        REAL,           -- MPa
    neat_nu12       REAL,
    neat_nu23       REAL,
    neat_rho        REAL,           -- kg/m³
    neat_CTE1       REAL,           -- 1/K  axial
    neat_CTE2       REAL,           -- 1/K  transverse
    neat_k1         REAL,           -- W/m·K  longitudinal
    neat_k2         REAL,           -- W/m·K  transverse
    neat_source     TEXT,
    neat_CTE_source TEXT,
    neat_k_source   TEXT,
    neat_notes      TEXT
);

CREATE TABLE IF NOT EXISTS polymers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    supplier        TEXT,
    neat_E1         REAL,           -- MPa
    neat_E2         REAL,           -- MPa
    neat_G12        REAL,           -- MPa
    neat_nu12       REAL,
    neat_rho        REAL,           -- kg/m³
    neat_CTE        REAL,           -- 1/K
    neat_k          REAL,           -- W/m·K
    neat_source     TEXT,
    neat_CTE_source TEXT,
    neat_k_source   TEXT,
    neat_notes      TEXT
);

CREATE TABLE IF NOT EXISTS printers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    manufacturer TEXT,
    notes        TEXT
);

-- ── material card tables ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS processing_conditions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    bead_width       REAL,           -- mm
    bead_height      REAL,           -- mm
    nozzle_diameter  REAL,           -- mm
    speed            REAL,           -- mm/min
    notes            TEXT,
    created_at       TEXT
);

CREATE TABLE IF NOT EXISTS print_configs (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    name                     TEXT NOT NULL,
    fiber_id                 INTEGER NOT NULL REFERENCES fibers(id),
    polymer_id               INTEGER NOT NULL REFERENCES polymers(id),
    printer_id               INTEGER          REFERENCES printers(id),
    processing_condition_id  INTEGER          REFERENCES processing_conditions(id),
    notes                    TEXT,
    created_at               TEXT
);

-- inference_runs is created before microstructure_snapshots because both
-- reference each other. SQLite allows forward FK references at DDL time.
CREATE TABLE IF NOT EXISTS inference_runs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    print_config_id        INTEGER NOT NULL REFERENCES print_configs(id),
    stage                  TEXT NOT NULL,
    -- 'elastic'|'thermoelastic'|'thermal_inverse'|'thermal_forward'
    microstructure_snap_id INTEGER,   -- REFERENCES microstructure_snapshots(id)
    inputs_json            TEXT,
    outputs_json           TEXT,
    solver_json            TEXT,
    loss                   REAL,
    notes                  TEXT,
    created_at             TEXT
);

CREATE TABLE IF NOT EXISTS microstructure_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    print_config_id  INTEGER NOT NULL REFERENCES print_configs(id),
    mf               REAL,           -- mass fraction (maps to w_f in model inputs)
    ar               REAL,
    a11              REAL,
    a22              REAL,
    a12              REAL,
    a13              REAL,
    a23              REAL,
    provenance_json  TEXT,           -- per-field {source_tag, inference_run_id}
    inference_run_id INTEGER REFERENCES inference_runs(id),
    notes            TEXT,
    created_at       TEXT
);

CREATE TABLE IF NOT EXISTS constituent_property_values (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    constituent_type TEXT NOT NULL,    -- 'fiber' | 'polymer'
    constituent_id   INTEGER NOT NULL, -- → fibers.id or polymers.id
    print_config_id  INTEGER REFERENCES print_configs(id),
    -- NULL = global (printer-agnostic, reusable across all cards)
    property_name    TEXT NOT NULL,
    value            REAL NOT NULL,
    unit             TEXT,
    source_tag       TEXT NOT NULL,    -- 'web'|'inputted'|'inferred'|'predicted'
    inference_run_id INTEGER REFERENCES inference_runs(id),
    notes            TEXT,
    created_at       TEXT
);

CREATE TABLE IF NOT EXISTS experimental_measurements (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    print_config_id  INTEGER NOT NULL REFERENCES print_configs(id),
    property_name    TEXT NOT NULL,
    value            REAL NOT NULL,
    unit             TEXT,
    uncertainty      REAL,            -- 1 std dev, same unit as value
    temperature_C    REAL,
    conditions_json  TEXT,
    reference        TEXT,
    date             TEXT,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS composite_property_values (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    print_config_id  INTEGER NOT NULL REFERENCES print_configs(id),
    property_name    TEXT NOT NULL,
    value            REAL NOT NULL,
    unit             TEXT,
    source_tag       TEXT NOT NULL,   -- 'predicted'|'experimental'|'inputted'
    inference_run_id INTEGER REFERENCES inference_runs(id),
    measurement_id   INTEGER REFERENCES experimental_measurements(id),
    temperature_C    REAL,
    created_at       TEXT
);

CREATE TABLE IF NOT EXISTS property_preferences (
    print_config_id  INTEGER NOT NULL REFERENCES print_configs(id),
    property_name    TEXT NOT NULL,
    preferred_source TEXT NOT NULL,
    -- 'experimental'|'predicted'|'inferred'|'web'|'inputted'
    PRIMARY KEY (print_config_id, property_name)
);

CREATE TABLE IF NOT EXISTS current_composite_properties (
    print_config_id  INTEGER NOT NULL,
    property_name    TEXT NOT NULL,
    value            REAL NOT NULL,
    unit             TEXT,
    source_tag       TEXT NOT NULL,
    source_run_id    INTEGER REFERENCES inference_runs(id),
    updated_at       TEXT,
    PRIMARY KEY (print_config_id, property_name)
);
"""



# ── seed helpers ──────────────────────────────────────────────────────────────

def _seed_fibers(conn: sqlite3.Connection):
    path = DATA_DIR / "fibers.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return
    fibers = json.loads(path.read_text())
    for f in fibers:
        neat = f.get("neat") or {}
        conn.execute(
            """
            INSERT INTO fibers
                (name, supplier,
                 neat_E1, neat_E2, neat_G12, neat_nu12, neat_nu23, neat_rho,
                 neat_CTE1, neat_CTE2, neat_k1, neat_k2,
                 neat_source, neat_CTE_source, neat_k_source, neat_notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                f.get("name"),
                f.get("supplier"),
                neat.get("E1"),
                neat.get("E2"),
                neat.get("G12"),
                neat.get("nu12"),
                neat.get("nu23"),
                neat.get("rho"),
                neat.get("CTE1"),
                neat.get("CTE2"),
                neat.get("k1"),
                neat.get("k2"),
                neat.get("source"),
                neat.get("CTE_source"),
                neat.get("k_source"),
                neat.get("notes"),
            ),
        )
    print(f"  seeded {len(fibers)} fiber(s)")


def _seed_polymers(conn: sqlite3.Connection):
    path = DATA_DIR / "polymers.json"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return
    polymers = json.loads(path.read_text())
    for p in polymers:
        neat = p.get("neat") or {}
        conn.execute(
            """
            INSERT INTO polymers
                (name, supplier,
                 neat_E1, neat_E2, neat_G12, neat_nu12, neat_rho,
                 neat_CTE, neat_k,
                 neat_source, neat_CTE_source, neat_k_source, neat_notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                p.get("name"),
                p.get("supplier"),
                neat.get("E1"),
                neat.get("E2"),
                neat.get("G12"),
                neat.get("nu12"),
                neat.get("rho"),
                neat.get("CTE"),
                neat.get("k"),
                neat.get("source"),
                neat.get("CTE_source"),
                neat.get("k_source"),
                neat.get("notes"),
            ),
        )
    print(f"  seeded {len(polymers)} polymer(s)")


# ── main ──────────────────────────────────────────────────────────────────────

def init(reset: bool = False):
    DATA_DIR.mkdir(exist_ok=True)

    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Deleted existing database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if reset:
        # drop in reverse dependency order
        for tbl in (
            "current_composite_properties",
            "property_preferences",
            "composite_property_values",
            "experimental_measurements",
            "constituent_property_values",
            "microstructure_snapshots",
            "inference_runs",
            "print_configs",
            "processing_conditions",
            "printers",
            "polymers",
            "fibers",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        conn.commit()

    conn.executescript(SCHEMA)
    conn.commit()

    print(f"Schema ready: {DB_PATH}")

    print("Seeding fibers …")
    _seed_fibers(conn)
    print("Seeding polymers …")
    _seed_polymers(conn)
    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    import sys
    reset = "--reset" in sys.argv or "-r" in sys.argv
    if not reset and DB_PATH.exists():
        ans = input(f"{DB_PATH} already exists. Reset and re-seed? [y/N] ").strip().lower()
        reset = ans == "y"
    init(reset=reset)
