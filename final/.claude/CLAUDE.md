# DiffMicromechanics — Project Context for Claude

This file captures what Claude knows about this project so future sessions start with full context.

---

## What This Project Is

A Python desktop application for **composite micromechanics** — predicting and inferring material properties of fiber-reinforced composites. It has three main capabilities:

1. **Forward prediction** — given fiber/matrix/microstructure inputs, predict composite elastic, thermoelastic, and thermal conductivity properties using surrogate neural network models.
2. **Inverse design (elastic/thermoelastic)** — given measured composite properties, infer the microstructure and constituent properties (matrix stiffness, fiber/matrix CTE) that produced them.
3. **Thermal inverse** — given measured composite thermal conductivity vs. temperature, recover constituent conductivities (k_f1, k_f2, k_m).

All three have GUI frontends (Tkinter) and the thermal inverse also has a CLI path.

---

## Entry Points

| Command | What it does |
|---|---|
| `python gui.py` | Forward prediction GUI |
| `python gui_inverse.py` | Inverse (elastic + thermoelastic) GUI |
| `python gui_thermal_inverse.py` | Thermal inverse GUI (also openable from gui_inverse.py) |
| `python scripts/run_inverse_thermal.py` | CLI thermal inverse, reads `config/thermal_problem.json` |
| `python db/init_db.py` | One-time setup: creates `data/micromechanics.db` and seeds material library |
| `python scripts/test_setup.py` | Verify the environment works |
| `python gui_material_card.py` | Standalone material card viewer |

All commands are run from the `final/` directory. Conda environment name: `diffmech`

---

## Directory Layout (post-refactor 2026-03-29)

```
final/
├── gui.py                    ← Forward prediction GUI
├── gui_inverse.py            ← Inverse (elastic + thermoelastic) GUI
├── gui_thermal_inverse.py    ← Thermal inverse GUI
├── gui_material_card.py      ← Material card viewer
├── gui_card_dialogs.py       ← Save/Load card dialogs
├── gui_identifiability.py    ← Identifiability analysis GUI
│
├── core/                     ← Computation / solver logic
│   ├── forward.py            ← Forward model loader (load_forward)
│   ├── inverse.py            ← Elastic/thermoelastic inverse solver
│   ├── inverse_thermal.py    ← Thermal inverse solver
│   ├── fim.py                ← Fisher Information Matrix utilities
│   ├── micro_surrogate.py    ← MICRO_SURROGATE_L2 and related classes
│   └── unit_manager.py       ← Unit conversion singleton (UM)
│
├── db/                       ← Database layer
│   ├── db.py                 ← All DB CRUD helpers
│   └── init_db.py            ← DB creation + seeding from JSON
│
├── config/                   ← Problem definitions + field labels
│   ├── field_labels.json
│   ├── problem.json
│   └── thermal_problem.json
│
├── scripts/                  ← CLI utilities and dev tools
│   ├── run_inverse_thermal.py
│   ├── test_setup.py
│   └── check_nu_sensitivity.py
│
├── NN_surrogate/             ← Neural network architecture (base classes)
├── models/                   ← Pre-trained model checkpoints
└── data/                     ← SQLite DB + seed JSON files
```

---

## Surrogate Models

- Located under `models/` (elastic, thermoelastic, thermal subdirectories)
- `NN_surrogate/` contains the base surrogate architecture (do not modify)
- `core/micro_surrogate.py` extends `NN_surrogate.models.SURROGATE` with MSE and L2 loss variants
- Models are checkpoint-based; loaded at runtime by `core/forward.py`

---

## Import Conventions

- GUI files import from `core.*`, `db.db`, and `NN_surrogate.*`
- Scripts in `scripts/` prepend `final/` to `sys.path` at startup so the same import paths work
- `db.db` is the single access point for the database — GUIs never call sqlite3 directly
- The `UM` singleton from `core.unit_manager` is shared across all GUI files

---

## Database System (SQLite — `data/micromechanics.db`)

### Core Design: Three-Layer Separation

| Layer | What it is | Printer-dependent? |
|---|---|---|
| Constituent | Fiber E, CTE, k / Matrix E, CTE, k | No — intrinsic material property |
| Microstructure | Orientation tensor (a11, a22…), mf, aspect ratio | Yes — printer determines fiber alignment |
| Composite | E1, CTE11, k11, … | Yes — derived from both layers |

### Key Tables

- **`fibers`** / **`polymers`** — static material library seeded from `data/fibers.json` / `data/polymers.json`. Never written to by inference.
- **`printers`** — printer registry (Markforged X7, Anisoprint, etc.)
- **`print_configs`** — a material card is a (fiber, polymer, printer) triple. Primary identity key.
- **`microstructure_snapshots`** — orientation + morphology snapshots, tagged with provenance (inputted vs. inferred)
- **`constituent_property_values`** — inferred values (matrix E, fiber CTE, k). `print_config_id=NULL` means global (applies to any card with that fiber/polymer).
- **`composite_property_values`** — predicted and measured composite properties
- **`experimental_measurements`** — user-measured values with uncertainty (σ)
- **`inference_runs`** — full audit trail of every solver run (inputs, outputs, loss, config JSON)
- **`property_preferences`** — user's preferred source per property per card

### Provenance Tags

Every stored value is tagged: `web` | `inputted` | `inferred` | `predicted`

### Unit Conventions

JSON seed files and the DB store in **SI/model units directly**: moduli in MPa, density in kg/m³, CTE in 1/K, conductivity in W/m·K. No unit conversion is needed when passing values to the surrogate.

---

## Current Build Status (as of 2026-03-29)

| Step | Status | Details |
|---|---|---|
| Seed JSON files | Done | `data/fibers.json`, `data/polymers.json` |
| DB creation + seeding | Done | `db/init_db.py` |
| DB helper module | Done | `db/db.py` |
| Forward GUI — material dropdowns | Done | Fiber/polymer auto-fill, Neat/In-situ toggle |
| Codebase refactor | Done | core/, db/, config/, scripts/ structure |
| Inverse GUI — Save to Card | Pending | After successful solve, save microstructure + inferred props |
| Material card viewer | Pending | `gui_material_card.py` shell exists |
| Thermal inverse — Save to Card | Pending | Phase 6 |

---

## Workflow Summary (Four Stages)

**Stage 0** — `python db/init_db.py` once. Optionally add printers via `db.db.add_printer(...)`.

**Stage 1 — Elastic inverse**: Fix fiber mechanical props from datasheet. Free matrix E/nu and all microstructure fields. Enter measured E1/E2/G12/nu12 as targets. Solve → Save to Card. Writes: microstructure snapshot, matrix E/nu as inferred constituent values, composite predictions, experimental measurements.

**Stage 2 — Thermoelastic inverse**: Load from Card (auto-fills Stage 1 microstructure + matrix E/nu as Fixed). Free only f_CTE1, f_CTE2, matrix_CTE. Enter measured CTE11/CTE22. Solve → Save to Card. CTE values stored globally (print_config_id=NULL — reusable across printers with same materials).

**Stage 3 — Thermal inverse**: Load k vs T CSV, run solver, save constituent k values (also global).

**Stage 4 — Forward prediction on new printer**: In gui.py, toggle In-situ → all previously inferred constituent props auto-fill. Enter new printer's orientation. Predict → Save prediction to a new card for Printer B.

---

## Seeded Materials

**Fibers**: Carbon Fiber T300 (Toray), E-Glass (Owens Corning), AS4
**Polymers**: PESU Ultrason (BASF), Epoxy 3501-6 (Hexcel)

---

## Notes

- The GUI uses Tkinter throughout — no web framework.
- `db/db.py` is the single access point for the database; GUIs never call sqlite3 directly.
- IDs are always resolved internally — users only ever see material names in dropdowns.
- `wandb/` run logs exist under `EL_surrogate/` and `TC_surrogate/` (surrogate training runs, not part of the app itself).
