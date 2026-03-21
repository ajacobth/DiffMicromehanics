"""init_db.py — create and seed data/micromechanics.db from the JSON seed files.

Run once (or re-run to reset):
    python init_db.py

Safe to re-run: drops and recreates all tables, then re-seeds from JSON.
No external dependencies — uses only Python stdlib (sqlite3, json, pathlib).
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
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    supplier     TEXT,
    neat_E1      REAL,          -- MPa
    neat_E2      REAL,          -- MPa
    neat_G12     REAL,          -- MPa
    neat_nu12    REAL,
    neat_nu23    REAL,
    neat_rho     REAL,          -- kg/m³
    neat_source  TEXT,
    neat_notes   TEXT
);

CREATE TABLE IF NOT EXISTS polymers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    supplier     TEXT,
    neat_E1      REAL,          -- MPa
    neat_E2      REAL,          -- MPa
    neat_G12     REAL,          -- MPa
    neat_nu12    REAL,
    neat_rho     REAL,          -- kg/m³
    neat_source  TEXT,
    neat_notes   TEXT
);

CREATE TABLE IF NOT EXISTS printers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    manufacturer TEXT,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS microstructure (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    fiber_id           INTEGER NOT NULL REFERENCES fibers(id),
    polymer_id         INTEGER NOT NULL REFERENCES polymers(id),
    printer_id         INTEGER          REFERENCES printers(id),
    Vf                 REAL,            -- volume fraction
    w_f                REAL,            -- weight fraction (nullable)
    ar                 REAL,            -- fiber aspect ratio
    a11                REAL,            -- orientation tensor components
    a22                REAL,
    a12                REAL,
    a13                REAL,
    a23                REAL,
    orientation_source TEXT,            -- e.g. "measured", "assumed", "CT scan"
    notes              TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    microstructure_id INTEGER NOT NULL REFERENCES microstructure(id),
    property_type     TEXT NOT NULL,   -- "elastic" | "thermoelastic" | "thermal"
    date              TEXT,            -- ISO-8601
    notes             TEXT,
    tags_json         TEXT             -- JSON string, e.g. {"machine":"CAMRI"}
);

CREATE TABLE IF NOT EXISTS inferred_properties (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    -- elastic
    E1            REAL,               -- MPa
    E2            REAL,               -- MPa
    G12           REAL,               -- MPa
    nu12          REAL,
    -- thermoelastic
    CTE1          REAL,               -- µ/K
    CTE2          REAL,               -- µ/K
    -- thermal
    TC1           REAL,               -- W/m·K
    TC2           REAL,               -- W/m·K
    -- solver metadata
    loss          REAL,
    model         TEXT,
    solver_json   TEXT                -- full solver config as JSON
);
"""


# ── helpers ───────────────────────────────────────────────────────────────────

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
                (name, supplier, neat_E1, neat_E2, neat_G12,
                 neat_nu12, neat_nu23, neat_rho, neat_source, neat_notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
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
                neat.get("source"),
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
                (name, supplier, neat_E1, neat_E2, neat_G12,
                 neat_nu12, neat_rho, neat_source, neat_notes)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                p.get("name"),
                p.get("supplier"),
                neat.get("E1"),
                neat.get("E2"),
                neat.get("G12"),
                neat.get("nu12"),
                neat.get("rho"),
                neat.get("source"),
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
        for tbl in ("inferred_properties", "experiments", "microstructure",
                    "printers", "polymers", "fibers"):
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
