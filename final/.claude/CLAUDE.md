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
| `python run_inverse_thermal.py` | CLI thermal inverse, reads `thermal_problem.json` |
| `python init_db.py` | One-time setup: creates `data/micromechanics.db` and seeds material library |
| `python test_setup.py` | Verify the environment works |
| `python gui_material_card.py` | Standalone material card viewer |

Conda environment name: `diffmech`

---

## Surrogate Models

- Located under `models/` (elastic, thermoelastic, thermal subdirectories)
- `NN_surrogate/` contains the surrogate model training/loading code
- Models are checkpoint-based; loaded at runtime by the GUI

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

## Key Source Files

| File | Purpose |
|---|---|
| `gui.py` | Forward prediction GUI — material library dropdowns (Fiber/Polymer, Neat/In-situ toggle) |
| `gui_inverse.py` | Inverse GUI — elastic + thermoelastic solve, "Save to Card" and "Load from Card" |
| `gui_thermal_inverse.py` | Thermal inverse GUI |
| `gui_material_card.py` | Material card viewer (read-only, all DB tables) |
| `gui_card_dialogs.py` | Dialog boxes for save/load card interactions |
| `db.py` | All DB helper functions (CRUD, unit conversions, card queries) |
| `models.py` | DB schema definitions |
| `init_db.py` | DB creation + seeding from JSON files |
| `forward.py` | Forward model wrapper |
| `inverse.py` | Inverse solver (elastic/thermoelastic) |
| `inverse_thermal.py` | Thermal inverse solver |
| `fim.py` | Fisher Information Matrix — identifiability analysis |

---

## Current Build Status (as of 2026-03-25)

| Step | Status | Details |
|---|---|---|
| Seed JSON files | Done | `data/fibers.json`, `data/polymers.json` |
| DB creation + seeding | Done | `init_db.py` |
| DB helper module | Done | `db.py` |
| Forward GUI — material dropdowns | Done | Fiber/polymer auto-fill, Neat/In-situ toggle |
| Inverse GUI — Save to Card | Pending | After successful solve, save microstructure + inferred props |
| Material card viewer | Pending | `gui_material_card.py` shell exists |
| Thermal inverse — Save to Card | Pending | Phase 6 |

---

## Workflow Summary (Four Stages)

**Stage 0** — `python init_db.py` once. Optionally add printers via `db.add_printer(...)`.

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
- `db.py` is the single access point for the database; GUIs never call sqlite3 directly.
- IDs are always resolved internally — users only ever see material names in dropdowns.
- `wandb/` run logs exist under `EL_surrogate/` and `TC_surrogate/` (surrogate training runs, not part of the app itself).
