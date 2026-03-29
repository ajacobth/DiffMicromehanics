# User Workflow — Material Card System
## Step-by-step guide: what you do, what gets saved, where

---

## Overview

The system has five distinct stages. You do not have to complete them all — each stage is independent and adds more data to the card. At any point you can open the Material Card Viewer to see what has been recorded.

```
Stage 0: First-time setup       → printers table
Stage 1: Elastic inverse        → microstructure_snapshots, constituent_property_values (matrix E/nu),
                                   inference_runs, composite_property_values
Stage 2: Thermoelastic inverse  → constituent_property_values (fiber CTE, matrix CTE),
                                   inference_runs, composite_property_values
Stage 3: Thermal inverse        → constituent_property_values (k_f1, k_f2, k_m),
                                   experimental_measurements (k vs T), inference_runs
Stage 4: Forward prediction     → composite_property_values (predicted), inference_runs
View:    Material Card Viewer   → read-only, all tables
```

---

## Stage 0 — First-time Setup (once per machine/environment)

**What you do:**
1. Run `python init_db.py` once. This creates `data/micromechanics.db` and seeds the fiber and polymer library.
2. Open `gui_inverse.py`. In the solver bar, click **"Load from Card"** (it will say no cards exist yet — ignore). Alternatively, run in a terminal:
   ```python
   import db
   db.add_printer("Markforged X7", manufacturer="Markforged")
   ```

**What happens in the DB:**

| Action | Table written | What is stored |
|---|---|---|
| `init_db.py` runs | `fibers` | Carbon Fiber T300, E-Glass, AS4 with neat mechanical + CTE + k properties |
| `init_db.py` runs | `polymers` | PESU Ultrason, Epoxy 3501-6 with neat properties |
| Add printer | `printers` | Printer name, manufacturer |

You only do this once. The fiber and polymer library never changes from inference — it is a static reference.

---

## Stage 1 — Elastic Inverse
**Goal:** Infer microstructure (a11, a22, ar, mf) and matrix stiffness (E, nu) from measured composite moduli.

**What you do:**
1. Open `python gui_inverse.py`
2. Select **Elastic** model, click **Load Model**
3. Set fiber mechanical properties (e1, e2, g12, f_nu12, f_nu23, fiber_density) to **Fixed** — these come from the datasheet. Optionally auto-fill them via the Material Library dropdowns.
4. Set matrix properties you know (e.g. matrix_density) to **Fixed**. Set `matrix_modulus` and `matrix_poisson` to **Free** (these will be inferred).
5. Set orientation fields (a11, a22, a12, a13, a23) and morphology fields (ar, fiber_massfrac) to **Free**.
6. In the Targets panel, check E1, E2, G12, nu12 and enter your measured values with σ (standard deviation from DMA/tensile tests).
7. Click **SOLVE**.
8. Click **"Save to Card"**:
   - Select fiber: `Carbon Fiber T300`
   - Select polymer: `PESU Ultrason`
   - Select printer: `Markforged X7`
   - Either select an existing card (to append) or type a name for a new one, e.g. `CF-PEI / MFX7`
   - Check "Save target values as experimental measurements" → YES (saves your measured E1/E2 with σ)
   - Click **Save**

**What gets written to the DB:**

| Table | Row(s) written | Content |
|---|---|---|
| `print_configs` | 1 new row (if creating new card) | Card name, fiber_id, polymer_id, printer_id |
| `inference_runs` | 1 row | stage='elastic', all inputs, all outputs, solver config, loss |
| `microstructure_snapshots` | 1 row | a11, a22, a12, a13, a23, ar, mf — all marked 'inferred' in provenance_json |
| `constituent_property_values` | 2 rows | matrix_modulus and matrix_poisson — source_tag='inferred', print_config_id=NULL (global) |
| `composite_property_values` | 9 rows | E1, E2, E3, G12, G13, G23, nu12, nu13, nu23 — source_tag='predicted' |
| `experimental_measurements` | 4 rows | E1, E2, G12, nu12 with uncertainty — your measured values |

**After this stage the card knows:** the microstructure for Printer A, and the in-situ matrix stiffness.

---

## Stage 2 — Thermoelastic Inverse
**Goal:** Infer fiber CTE and matrix CTE from measured composite thermal expansion. Microstructure is fixed from Stage 1.

**What you do:**
1. Still in `gui_inverse.py`, select **Thermoelastic** model, click **Load Model**
2. Click **"Load from Card"** → select `CF-PEI / MFX7`
   - This auto-fills a11, a22, a12, a13, a23, ar, mf (from the Stage 1 microstructure snapshot) as **Fixed**
   - It also fills matrix_modulus and matrix_poisson (from Stage 1 inferred values) as **Fixed**
   - All fiber mechanical properties are pre-filled from the fibers table as **Fixed**
3. Change only `f_CTE1`, `f_CTE2`, and `matrix_CTE` to **Free** — everything else stays Fixed
4. In Targets, check CTE11 and CTE22, enter your measured dilatometry values with σ
5. Click **SOLVE** → Click **"Save to Card"** → pick the same card

**What gets written:**

| Table | Row(s) written | Content |
|---|---|---|
| `inference_runs` | 1 row | stage='thermoelastic', full inputs/outputs, loss |
| `microstructure_snapshots` | 1 row | Same microstructure values — all fields marked 'inputted' (nothing new was inferred about microstructure) |
| `constituent_property_values` | 3 rows | f_CTE1 (fiber), f_CTE2 (fiber), matrix_CTE (polymer) — **print_config_id=NULL** (global — reusable across all cards with this fiber+polymer) |
| `composite_property_values` | 15 rows | All thermoelastic outputs — source_tag='predicted' |
| `experimental_measurements` | 2 rows | CTE11, CTE22 with uncertainty |

**Key point:** `print_config_id=NULL` means fiber CTE and matrix CTE are stored globally — they are intrinsic material properties, independent of which printer was used. Any future card with the same CF-PEI materials automatically picks them up.

**After this stage the card knows:** complete thermoelastic microstructure and constituent properties.

---

## Stage 3 — Thermal Inverse (coming in Phase 6)
**Goal:** Infer constituent thermal conductivities from measured composite k vs. temperature.

**What you do:**
1. In `gui_inverse.py`, click **"Open Thermal Inverse Solver"**
2. Load your k11/k22/k33 vs T CSV
3. Run the solver
4. *(Phase 6)* Click "Save to Card"

**What gets written:**

| Table | Row(s) written | Content |
|---|---|---|
| `experimental_measurements` | N rows (one per T per direction) | k11, k22, k33 at each temperature with uncertainty |
| `inference_runs` | 1 row | stage='thermal_inverse', parametric outputs (p1, p2, l2, t or k_f1, k_f2, k_m) |
| `constituent_property_values` | 2–4 rows | k_f1, k_f2 (fiber), k_m (polymer) — all print_config_id=NULL (global) |

---

## Stage 4 — Forward Prediction from Card
**Goal:** Predict composite properties for a second printer (Printer B) using constituent properties already known from Stages 1–3.

**What you do:**
1. Open `python gui.py`
2. Select a model (e.g. Thermoelastic), click **Load Model**
3. Select Fiber: `Carbon Fiber T300` and Polymer: `PESU Ultrason` from the dropdowns
4. Toggle **"In-situ"** — all fields that were inferred in Stages 1–2 auto-fill (matrix_modulus, matrix_poisson, f_CTE1, f_CTE2, matrix_CTE). Fiber mechanical properties fill from the datasheet.
5. Enter the orientation (a11, a22 etc.) and mass fraction for Printer B manually — these are Printer B specific
6. Click **Predict**
7. Click **"Save Prediction to Card"** → select or create card for `CF-PEI / Anisoprint`

**What gets written:**

| Table | Row(s) written | Content |
|---|---|---|
| `print_configs` | 1 new row (if new card) | CF-PEI / Anisoprint |
| `microstructure_snapshots` | 1 row | Printer B orientation — all fields marked 'inputted' |
| `inference_runs` | 1 row | stage='thermoelastic_forward', all inputs/outputs |
| `composite_property_values` | N rows | All predicted outputs — source_tag='predicted' |

**Key insight:** The constituent properties (matrix stiffness, fiber CTE) were inferred on Printer A (Stage 1–2) but are globally stored. Printer B's card immediately benefits from them — you did not need to run the inverse solver for Printer B.

---

## Viewing the Material Card

Click **"Material Card Viewer"** in the top bar of `gui.py`, or run:
```bash
python gui_material_card.py
```

**Tabs:**

| Tab | What you see |
|---|---|
| Summary | Card metadata, fiber/polymer/printer names, microstructure snapshot, row counts |
| Constituent Properties | Side-by-side: neat (web) value vs. inferred value, source tag, which run produced it, when |
| Microstructure | All snapshots with per-field provenance (inputted vs. inferred), full history |
| Composite Properties | All predicted and experimental composite properties with temperature and uncertainty |
| Inference History | All runs, click a row to expand full inputs/outputs JSON |

---

## Summary: Which Table for What?

| Data type | Table | print_config_id? |
|---|---|---|
| Static material library (fiber/polymer specs) | `fibers`, `polymers` | N/A — global reference |
| Printer registry | `printers` | N/A |
| Card identity (fiber + polymer + printer) | `print_configs` | IS the key |
| Orientation/morphology snapshot | `microstructure_snapshots` | Required — printer-specific |
| Inferred constituent property (E, CTE, k) | `constituent_property_values` | NULL = global, or specific = card-scoped |
| Predicted composite property | `composite_property_values` source_tag='predicted' | Required |
| Measured composite property | `experimental_measurements` | Required |
| Preferred source per property | `property_preferences` | Required |
| Audit trail of every solve | `inference_runs` | Required |

---

## ID Resolution — You Never See IDs

Every dropdown in the GUIs shows **names** (e.g. "Carbon Fiber T300"). The fiber_id, polymer_id, printer_id, and print_config_id are resolved internally:

```
User picks: "Carbon Fiber T300"
GUI does:   fiber_id = fiber_map["Carbon Fiber T300"]  = 1
DB stores:  fiber_id = 1
```

When viewing a card, IDs are translated back to names. You never need to know or type an ID.
