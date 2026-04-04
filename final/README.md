# DiffMicromechanics

A Python desktop application for composite micromechanics — predicting and
inferring material properties of fiber-reinforced composites using surrogate
neural network models.

---

## What It Does

Three core capabilities:

| Capability | Entry point | Description |
|---|---|---|
| **Forward prediction** | `python gui.py` | Given microstructure + constituent inputs, predict composite elastic, thermoelastic, and thermal conductivity properties |
| **Inverse design** | `python gui_inverse.py` | Given measured composite properties, infer the microstructure and constituent properties that produced them |
| **Thermal inverse** | `python gui_thermal_inverse.py` | Given measured composite k vs T data, recover constituent conductivities (k_f1, k_f2, k_m) |

The key capability that ties all three together is the **material card** — a
persistent record of a (fiber + polymer + printer) combination that accumulates
characterization results across stages and enables property transfer to new
printers without re-running experiments.

---

## Material Card and Property Transfer

The material card system is the central feature of this application.

### What a material card is

A material card is identified by a **(fiber, polymer, printer)** triple stored
in `print_configs`. It accumulates results from each characterization stage:

```
Stage 1 — Elastic inverse
  → stores: microstructure snapshot (a11, a22, ar, mf, ...)
            matrix modulus + poisson (inferred)

Stage 2 — Thermoelastic inverse
  → stores: fiber CTE, matrix CTE (inferred, globally reusable)

Stage 3 — Thermal inverse
  → stores: k_f1, k_f2, k_m (inferred, globally reusable)

Stage 4 — Forward prediction on a new printer
  → stores: predicted composite properties for Printer B
```

### Property transfer to a new printer

Once Stages 1–3 are complete on Printer A, transferring to Printer B requires
**no new inverse solves and no new material experiments**:

1. Open `gui.py`
2. Select the same fiber and polymer
3. Toggle **In-situ** — all inferred constituent properties (matrix modulus,
   fiber/matrix CTE, conductivities) auto-fill from the database
4. Enter Printer B's orientation tensor (a11, a22, a12, a13, a23, ar, mf)
5. Click **Predict** → save to a new card for Printer B

The constituent properties are stored globally (`print_config_id = NULL`)
because they are intrinsic to the fiber and polymer — they do not change with
the printer. Only the microstructure (orientation, mass fraction) is
printer-specific. This means one set of inverse experiments characterises a
fiber-polymer pair for all printers that use those materials.

### Why this matters

| Without material cards | With material cards |
|---|---|
| Re-run experiments for every new printer | Characterise once, transfer to any printer |
| Manually copy inferred values between GUIs | In-situ toggle auto-fills from DB |
| No audit trail | Full inference history with inputs, outputs, loss |
| Properties siloed per run | Global constituent properties shared across cards |

---

## Directory Structure

```
final/
│
├── gui.py                    ← Forward prediction GUI
├── gui_inverse.py            ← Inverse (elastic + thermoelastic) GUI
├── gui_thermal_inverse.py    ← Thermal inverse GUI
├── gui_material_card.py      ← Material card viewer (standalone)
├── gui_card_dialogs.py       ← Save to Card / Load from Card dialogs
│                               (shared between gui.py and gui_inverse.py)
├── gui_identifiability.py    ← Identifiability analysis GUI
│
├── core/                     ← Computation and solver logic
│   ├── forward.py            ← Forward model loader
│   ├── inverse.py            ← Elastic / thermoelastic inverse solver
│   ├── inverse_thermal.py    ← Thermal inverse solver
│   ├── fim.py                ← Fisher Information Matrix utilities
│   ├── micro_surrogate.py    ← Surrogate model class extensions
│   └── unit_manager.py       ← Unit conversion singleton (UM)
│
├── db/                       ← Database layer
│   ├── db.py                 ← All DB CRUD helpers (single access point)
│   └── init_db.py            ← DB creation + seeding from JSON
│
├── config/                   ← Problem definitions and field labels
│   ├── field_labels.json     ← Human-readable labels for all model fields
│   ├── problem.json          ← Elastic / thermoelastic problem config
│   └── thermal_problem.json  ← Thermal inverse problem config
│
├── scripts/                  ← CLI utilities and dev tools
│   ├── run_inverse_thermal.py  ← CLI thermal inverse (reads thermal_problem.json)
│   ├── test_setup.py           ← Environment verification script
│   └── check_nu_sensitivity.py ← Developer utility
│
├── models/                   ← Pre-trained surrogate model checkpoints
│   ├── elastic/
│   ├── thermoelastic/
│   └── thermal/
│
├── NN_surrogate/             ← Neural network base architecture (do not modify)
│
└── data/                     ← SQLite database + seed JSON files
    ├── micromechanics.db     ← Created by db/init_db.py
    ├── fibers.json           ← Seed data for fiber library
    └── polymers.json         ← Seed data for polymer library
```

---

## Quick Reference

All commands are run from the `final/` directory with the `diffmech` conda
environment active.

```bash
conda activate diffmech
cd path/to/DiffMicromehanics/final

# One-time setup: create and seed the database
python db/init_db.py

# Verify the environment
python scripts/test_setup.py

# Forward prediction GUI
python gui.py

# Inverse design GUI (elastic + thermoelastic + opens thermal inverse)
python gui_inverse.py

# Thermal inverse GUI (standalone)
python gui_thermal_inverse.py

# Material card viewer (standalone)
python gui_material_card.py

# Thermal inverse from the command line
python scripts/run_inverse_thermal.py
python scripts/run_inverse_thermal.py --problem config/thermal_problem.json
python scripts/run_inverse_thermal.py --problem p.json --output_dir results/
```

---

## Database Design (Three-Layer Separation)

| Layer | Table(s) | Printer-dependent? | Description |
|---|---|---|---|
| Constituent | `fibers`, `polymers`, `constituent_property_values` | No | Intrinsic material properties — shared across all printers |
| Microstructure | `microstructure_snapshots` | Yes | Fiber orientation and morphology — specific to printer |
| Composite | `composite_property_values`, `experimental_measurements` | Yes | Derived from both layers above |

**Provenance tags** on every stored value: `web` \| `inputted` \| `inferred` \| `predicted`

**Unit convention**: all values stored in model units — moduli in MPa,
density in kg/m³, CTE in 1/K, conductivity in W/m·K.

---

## Seeded Material Library

**Fibers**: Carbon Fiber T300 (Toray), E-Glass (Owens Corning), AS4

**Polymers**: PESU Ultrason (BASF), Epoxy 3501-6 (Hexcel)

Add new fibers, polymers, and printers via the **Manage Materials** button
in either GUI.

---

## Further Reading

- `SETUP_AND_RUN.md` — full environment setup and step-by-step GUI walkthrough
- `USER_WORKFLOW.md` — detailed stage-by-stage DB write log
- `DATABASE_PLAN.md` — full schema documentation
- `MATERIAL_CARD_PLAN.md` — material card system design
- `.claude/AGENT_PLAN.md` — planned agentic interface (LangGraph)
- `.claude/viscoelastic_db_plan.md` — planned viscoelastic extension
