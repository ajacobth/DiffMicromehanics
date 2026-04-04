# Transfer Window — Implementation Plan

## Purpose

Allow a user to take a material (fiber + polymer) that has been characterized on **Printer A**
(inferred constituent properties stored in the DB) and produce a forward prediction for the
same material on **Printer B** — either by entering the new printer's microstructure manually
or by inferring it from new measurements.

---

## Architecture

**New file:** `gui_transfer.py` — `TransferWindow(tk.Toplevel)`

A standalone top-level window (not a modal dialog) opened via a
**"Transfer to New Printer…"** button added to `gui.py`'s top bar.
Single scrollable window with three sequential sections that unlock in order.

---

## Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Transfer Material to New Printer                           │
│─────────────────────────────────────────────────────────────│
│  ① Source Card                                              │
│     Card: [ CF T300 / PESU / Markforged X7 ▼ ]             │
│     Constituent props found:                                │
│       matrix_modulus  3410 MPa   [inferred]                 │
│       f_CTE1          -0.5e-6    [inferred]                 │
│       f_CTE2           15e-6     [inferred]                 │
│       (missing: k_f1, k_f2, k_m — thermal not yet run)     │
│─────────────────────────────────────────────────────────────│
│  ② Microstructure for New Printer                           │
│     ○ Infer from measurements    ● Enter manually           │
│                                                             │
│  [Manual mode]                                              │
│     a11: [  ]  a22: [  ]  a12: [  ]                        │
│     a13: [  ]  a23: [  ]  ar:  [  ]  mf: [  ]             │
│                                                             │
│  [Infer mode]                                               │
│     ☑ E1:  [  ] ±σ[  ]   ☑ E2:  [  ] ±σ[  ]              │
│     ☑ G12: [  ] ±σ[  ]   ☑ nu12:[  ] ±σ[  ]              │
│     [ Run Inverse ]   status: …                             │
│                                                             │
│  Locked constituent inputs (read-only, from source card):  │
│     e1 = 230000 MPa   matrix_modulus = 3410 MPa  …        │
│─────────────────────────────────────────────────────────────│
│  ③ Predict & Save                                           │
│     Model: ● Elastic  ○ Thermoelastic  ○ Thermal           │
│     [ Predict ]    E1 = … MPa   CTE11 = … 1/K  …          │
│                                                             │
│     New Printer: [ Printer B ▼ ]  [ + Add printer… ]       │
│     Card name:   [ CF/PESU/PrinterB-Transfer ]             │
│     [ Save to New Card ]                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
Source card (fiber_id, polymer_id)
    ↓
get_constituent_properties("fiber",   fid, include_global=True)  → locked fiber props
get_constituent_properties("polymer", pid, include_global=True)  → locked polymer props
fiber_model_inputs(fid) + polymer_model_inputs(pid)              → neat base (fallback)
    ↓
Merge: inferred constituent overrides on top of neat base
    ↓
Microstructure (entered directly OR solved by inline inverse)
    ↓
Forward predict  →  predicted_outputs dict
    ↓
SaveToCardDialog(result, fiber_id=fid, polymer_id=pid)
```

---

## Key Design Decisions

### 1. Constituent props are always locked (read-only)
Transfer assumes constituent properties are material-specific and fixed.
Show them as a read-only grid. If any are missing (e.g. thermal inverse not yet run),
show a warning badge listing which properties fall back to neat datasheet values.

### 2. Microstructure mode is a radio pair, not a separate window
The window swaps the middle section in-place between:
- **Manual** — direct entry fields for a11, a22, a12, a13, a23, ar, mf
- **Infer** — target output fields (E1, E2, G12, nu12 with σ) + "Run Inverse" button

No extra dialogs. State stays visible throughout.

### 3. Inverse mode calls `core.inverse` directly, not `gui_inverse.py`
`InverseGUI` is too heavy to embed. `TransferWindow` calls
`core.inverse.run_inverse(...)` in a thread (same pattern as
`gui_inverse.py`'s `_on_solve`) with a minimal target-field UI.
After the solve, the inferred microstructure fields are written into
the manual-entry widgets so the user can review/edit before predicting.

### 4. Save delegates entirely to `SaveToCardDialog`
No new save logic. `SaveToCardDialog` already handles writing:
- `microstructure_snapshots`
- `composite_property_values` (predicted)
- `inference_runs` (if inverse was used)
Pre-populate `fiber_id` and `polymer_id` from the source card.
The user selects only the new printer in the dialog.

### 5. Constituent properties are NOT re-written on save
They are already in the DB as global (`print_config_id=NULL`) rows from
the original Printer A characterization. The new card reuses them automatically
via `get_constituent_properties(..., include_global=True)`.

---

## Files to Change

| File | Change |
|---|---|
| `gui_transfer.py` | **New** — `TransferWindow` class |
| `gui.py` | Add "Transfer to New Printer…" button in top bar; add `_open_transfer()` method |
| `gui_inverse.py` | Add same button (lower priority / optional) |

`gui_card_dialogs.py` and `db/db.py` — **no changes needed**.
All required helpers already exist.

---

## Implementation Order

1. `gui_transfer.py` — section ① source card selector + constituent props summary table
2. Section ② manual microstructure entry + locked-inputs read-only grid
3. Section ③ forward predict + `SaveToCardDialog` call
4. Section ② infer mode — inline target fields + "Run Inverse" button + thread
5. Wire "Transfer to New Printer…" button into `gui.py` top bar

---

## DB Helpers Used (all already exist in `db/db.py`)

| Helper | Used for |
|---|---|
| `get_all_print_configs()` | Populate source card dropdown |
| `get_print_config(id)` | Resolve fiber_id / polymer_id from selected card |
| `fiber_model_inputs(fid)` | Neat fiber base props |
| `polymer_model_inputs(pid)` | Neat polymer base props |
| `get_constituent_properties(type, id, include_global=True)` | Inferred overrides |
| `get_latest_microstructure(card_id)` | Optional: pre-fill micro fields from source card |

---

## Status

- [ ] Section ① — source card selector
- [ ] Section ② — manual mode
- [ ] Section ③ — predict + save
- [ ] Section ② — infer mode
- [ ] Button wired into `gui.py`
