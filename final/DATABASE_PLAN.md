# DiffMicromechanics — Database Integration Plan

## Data flow

```
fibers.json / polymers.json
         │
    init_db.py  ──►  data/micromechanics.db
                           │
              ┌────────────┼─────────────────┐
              ▼            ▼                 ▼
        forward GUI   inverse GUI      material card
              │            │
    load fiber/polymer  solve → save experiment
    → auto-fill inputs  → write inferred_properties
                        → link to fiber/polymer used
```

---

## Database schema (SQLite — no pip installs needed)

### `fibers`
| column | type | notes |
|---|---|---|
| id | INTEGER PK | |
| name | TEXT | |
| supplier | TEXT | |
| neat_E1 | REAL | GPa |
| neat_E2 | REAL | GPa |
| neat_G12 | REAL | GPa |
| neat_nu12 | REAL | |
| neat_rho | REAL | g/cm³ |
| neat_source | TEXT | |
| neat_notes | TEXT | |

### `polymers`
Same structure as `fibers` (neat_* columns, same units).

### `printers`
| column | type |
|---|---|
| id | INTEGER PK |
| name | TEXT |
| manufacturer | TEXT |
| notes | TEXT |

### `microstructure`
Links a fiber + polymer + printer combination with processing parameters.

| column | type | notes |
|---|---|---|
| id | INTEGER PK | |
| fiber_id | INTEGER FK | → fibers.id |
| polymer_id | INTEGER FK | → polymers.id |
| printer_id | INTEGER FK | → printers.id, nullable |
| Vf | REAL | volume fraction |
| w_f | REAL | weight fraction, nullable |
| ar | REAL | fiber aspect ratio |
| a11 | REAL | orientation tensor |
| a22 | REAL | orientation tensor |
| a12 | REAL | orientation tensor |
| a13 | REAL | orientation tensor |
| a23 | REAL | orientation tensor |
| orientation_source | TEXT | e.g. "measured", "assumed", "CT scan" |
| notes | TEXT | |

### `experiments`
One row per inverse-solver run that produced inferred properties.

| column | type | notes |
|---|---|---|
| id | INTEGER PK | |
| microstructure_id | INTEGER FK | → microstructure.id |
| property_type | TEXT | "elastic", "thermoelastic", "thermal" |
| date | TEXT | ISO-8601 |
| notes | TEXT | |
| tags_json | TEXT | JSON string, e.g. {"machine":"CAMRI"} |

### `inferred_properties`
Stores the output of one inverse solve.

| column | type | notes |
|---|---|---|
| id | INTEGER PK | |
| experiment_id | INTEGER FK | → experiments.id |
| E1 | REAL | GPa, nullable |
| E2 | REAL | GPa, nullable |
| G12 | REAL | GPa, nullable |
| nu12 | REAL | nullable |
| CTE1 | REAL | µ/K, nullable |
| CTE2 | REAL | µ/K, nullable |
| TC1 | REAL | W/m·K, nullable |
| TC2 | REAL | W/m·K, nullable |
| loss | REAL | final optimiser error |
| model | TEXT | e.g. "elastic" |
| solver_json | TEXT | full solver config as JSON |

---

## Material card concept

For a given fiber + polymer pair, the card shows:

| Property | Neat (datasheet) | Best in-situ (inferred) | Source |
|---|---|---|---|
| E1 | 230 GPa | — | Toray datasheet |
| E2 | 15 GPa | 13.2 GPa | Exp #12, 2026-03-15, loss=0.003 |

In-situ values come from the lowest-loss `inferred_properties` record for that fiber/polymer combination. The card also lists all experiments that used this pair.

---

## Unit conventions

JSON seed files and database store properties in **display units**:
- Moduli: GPa
- Density: g/cm³
- CTE: µ/K
- Conductivity: W/m·K

`db.py` load functions apply unit conversions to match model input units (MPa, kg/m³, 1/K) before returning values for GUI auto-fill.

---

## Build order

| Step | Status | What | Files |
|---|---|---|---|
| 1 | ✅ done | Seed JSON files | `data/fibers.json`, `data/polymers.json` |
| 2 | ✅ done | DB creation + seeding | `init_db.py` |
| 3 | ✅ done | DB helper module | `db.py` |
| 4 | ✅ done | Forward GUI — material dropdowns | `gui.py` |
| 5 | pending | Inverse GUI — save-to-DB after solve | `gui_inverse.py` |
| 6 | pending | Material card viewer | `gui_material_card.py` |

---

## Step 4 — Forward GUI additions ✅

A "Material Library" bar was added between the top controls and the inputs/outputs panel.

**What was built:**
- **Fiber dropdown** — selecting a fiber auto-fills matching model input fields:
  `e1, e2, g12, f_nu12, fiber_density, rho_f` (unit conversion: GPa→MPa, g/cm³→kg/m³)
- **Polymer dropdown** — auto-fills:
  `matrix_modulus, matrix_poisson, matrix_density, rho_m`
- **Source toggle: Neat | In-situ** — In-situ is greyed out until inverse solve data exists for that fiber/polymer pair; enables automatically once it does
- Dropdowns are read-only (not editable); selecting one fills fields but you can still type over any value manually
- If `init_db.py` has not been run, a red warning message appears and dropdowns stay disabled
- Switching surrogate models (e.g. elastic → thermal) and re-loading re-applies any selected materials to the new input fields

**Testing checklist:**
- [ ] Run `python gui.py` — library bar appears below the top controls
- [ ] Library status shows "Library: 2 fibers, 2 polymers"
- [ ] Load a model (e.g. elastic), select Carbon Fiber T300 → `e1`, `e2`, `g12`, `f_nu12`, `fiber_density` fields fill
- [ ] Select PESU Ultrason → `matrix_modulus`, `matrix_poisson`, `matrix_density` fields fill
- [ ] Manually edit a filled field — value stays changed (no override)
- [ ] In-situ radio button is greyed out (no inference data yet)
- [ ] Selecting "— none —" from dropdown does not clear fields

---

## Step 5 — Inverse GUI additions (pending)

After a successful solve, enable a **"Save to Database"** button that opens a dialog:
- Pick fiber and polymer used in the experiment
- Pick printer (optional)
- Enter Vf, aspect ratio, orientation source, notes
- On confirm: upsert `microstructure` → insert `experiment` → insert `inferred_properties`
- After saving, the In-situ toggle in the forward GUI will become active for that fiber/polymer pair

## Step 6 — Material card viewer (pending)

Standalone window (`gui_material_card.py`) launched from either GUI:
- Fiber/polymer selector at top
- Neat properties table
- In-situ properties table (best-loss record highlighted)
- Experiment history list (date, property_type, loss, tags)
- "Use these values" button to push in-situ props back to the calling GUI
