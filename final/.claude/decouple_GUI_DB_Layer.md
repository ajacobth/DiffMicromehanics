# Plan: Decouple GUI from DB/Solver Layer

## Goal

Introduce a `material_service.py` layer that sits between any GUI and the
DB/solver code. After this refactor, any frontend — Tkinter, FastAPI, Streamlit,
CLI, or AI agent — calls the same Python functions with plain dicts in and plain
dicts out. No frontend code touches `db.py`, `forward.py`, or `inverse.py` directly.

---

## Why this is the right time

- `db.py`, `forward.py`, `inverse.py`, `inverse_thermal.py` are already GUI-free
- The business logic that needs to move is isolated to three functions in two files
- The Tkinter GUI currently works; we are thinning it, not rewriting it

---

## Current layer map (what lives where today)

| Logic | Currently lives in | Should live in |
|---|---|---|
| Solver orchestration + result dict assembly | `gui_inverse._solve_worker` (lines 691–761) | `material_service.run_inverse()` |
| 6-step DB write sequence after a solve | `gui_card_dialogs.SaveToCardDialog._on_save` (lines 356–547) | `material_service.save_to_card()` |
| Two-pass inferred > inputted load + neat fallback | `gui_card_dialogs.LoadFromCardDialog._on_load` (lines 693–811) | `material_service.load_card_inputs()` |
| Forward prediction orchestration | `gui.py` predict callback | `material_service.predict()` |
| Thermal inverse solver orchestration | `gui_thermal_inverse.py` solve worker | `material_service.run_thermal_inverse()` |
| Thermal save sequence | `gui_card_dialogs.SaveThermalToCardDialog._on_save` | `material_service.save_thermal_to_card()` |

`db.py`, `forward.py`, `inverse.py`, `inverse_thermal.py` — **untouched throughout**.

---

## Target architecture

```
Tkinter GUI        FastAPI + React      Streamlit      CLI / Agent
      ↓                   ↓                 ↓               ↓
                  material_service.py    (new)
          ┌──────────────────────────────────────────┐
          │  predict()                               │
          │  run_inverse()                           │
          │  run_thermal_inverse()                   │
          │  load_card_inputs()                      │
          │  save_to_card()                          │
          │  save_thermal_to_card()                  │
          └──────────────────────────────────────────┘
                    ↓                   ↓
                 db.py       forward.py / inverse.py / inverse_thermal.py
                    ↓
                 SQLite
```

---

## Data contracts (new types)

Create `service_types.py` with these dataclasses. No behaviour — just the
contracts that every frontend and the service layer agree on.

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass
class InverseResult:
    """Output of material_service.run_inverse()."""
    model:             str
    free_variables:    dict[str, float]   # inferred values
    fixed_inputs:      dict[str, float]
    predicted_outputs: dict[str, float]
    target_outputs:    dict[str, float]
    sigmas:            dict[str, float]
    loss:              float
    solver_cfg:        dict

@dataclass
class PredictionResult:
    """Output of material_service.predict()."""
    model:   str
    inputs:  dict[str, float]
    outputs: dict[str, float]

@dataclass
class ThermalInverseResult:
    """Output of material_service.run_thermal_inverse()."""
    k_f1:         float
    k_f2:         float
    k_m:          float
    parametric:   dict[str, float]   # p1, p2, l2, t
    temperatures: np.ndarray
    K_pred:       np.ndarray         # shape (n_temps, 3) — K11, K22, K33
    loss:         float
    solver_cfg:   dict

@dataclass
class CardInputs:
    """Output of material_service.load_card_inputs(). Ready to paste into any GUI."""
    microstructure: dict[str, float]  # a11, a22, a12, a13, a23, mf, ar
    fiber_props:    dict[str, float]  # inferred > inputted > neat, keyed by model field name
    polymer_props:  dict[str, float]
    provenance:     dict[str, str]    # field → "inferred" | "inputted" | "neat"
    fiber_id:       int
    polymer_id:     int
    card_id:        int
```

---

## Service layer API (`material_service.py`)

```python
# ── Reference data ────────────────────────────────────────────────────────────
# These are thin wrappers around db.py for callers who want a single import.
def list_fibers()   -> list[dict]: ...
def list_polymers() -> list[dict]: ...
def list_printers() -> list[dict]: ...
def add_printer(name: str, manufacturer: str = "") -> int: ...

# ── Forward prediction ────────────────────────────────────────────────────────
def predict(
    fiber_id:      int,
    polymer_id:    int,
    microstructure: dict[str, float],
    model:         str = "elastic",
) -> PredictionResult: ...

# ── Elastic / thermoelastic inverse ──────────────────────────────────────────
def run_inverse(
    fiber_id:       int,
    polymer_id:     int,
    fixed_inputs:   dict[str, float],
    free_fields:    list[str],
    bounds:         dict[str, tuple[float, float]],
    target_outputs: dict[str, float],
    sigmas:         dict[str, float],
    model:          str = "elastic",
    solver_cfg:     dict | None = None,
) -> InverseResult: ...

# ── Thermal inverse ───────────────────────────────────────────────────────────
def run_thermal_inverse(
    fiber_id:       int,
    polymer_id:     int,
    fixed_inputs:   dict[str, float],
    temperatures:   list[float],
    k_measurements: dict[str, list[float]],  # {"K11": [...], "K22": [...], ...}
    solver_cfg:     dict | None = None,
) -> ThermalInverseResult: ...

# ── Card operations ───────────────────────────────────────────────────────────
def load_card_inputs(card_id: int) -> CardInputs: ...

def save_to_card(
    result:            InverseResult | PredictionResult,
    fiber_id:          int,
    polymer_id:        int,
    card_id:           int | None,   # None → create new card
    card_name:         str = "",
    printer_id:        int | None = None,
    save_experimental: bool = True,
    notes:             str = "",
) -> int: ...   # returns card_id

def save_thermal_to_card(
    result:     ThermalInverseResult,
    fiber_id:   int,
    polymer_id: int,
    card_id:    int | None,
    card_name:  str = "",
    printer_id: int | None = None,
    notes:      str = "",
) -> int: ...
```

---

## Implementation phases

Each phase is independently runnable. After every phase, the full GUI must still
work — verify by running the EVALUATION_PLAN.md sections relevant to what changed.

---

### Phase 1 — Create `service_types.py`

**Files created:** `service_types.py`
**Files changed:** none

Write the four dataclasses from the Data Contracts section above.
No imports from `tkinter`, `db`, or `forward` — just `dataclasses` and `numpy`.

**Verification:**
```bash
python -c "from service_types import InverseResult, PredictionResult, ThermalInverseResult, CardInputs; print('OK')"
```
Must print `OK` with no errors.

---

### Phase 2 — Extract `load_card_inputs()`

**Files created:** `material_service.py` (initial)
**Files changed:** `gui_card_dialogs.py`

Move the logic from `LoadFromCardDialog._on_load` (lines 693–811) into
`material_service.load_card_inputs(card_id: int) -> CardInputs`.

This includes:
- The microstructure snapshot read
- The two-pass inferred > inputted resolution loop (`inferred_best`, `inputted_best`)
- The neat fallback maps (`_FIBER_NEAT_MAP`, `_POLY_NEAT_MAP`)

After extraction, `_on_load` becomes:
```python
def _on_load(self):
    card = self._card_map.get(self._card_var.get())
    if not card:
        messagebox.showerror(...)
        return
    try:
        inputs = material_service.load_card_inputs(card["id"])
    except Exception as exc:
        messagebox.showerror("Load Error", str(exc), parent=self._win)
        return
    self.loaded = {**inputs.microstructure, **inputs.fiber_props, **inputs.polymer_props}
    self.loaded_provenance = inputs.provenance
    self.loaded_card = card
    self._win.destroy()
```

**Verification:** EVALUATION_PLAN.md Section 3 — Load from Card still populates
all fields correctly. Section 6.4 round-trip still passes.

---

### Phase 3 — Extract `save_to_card()`

**Files changed:** `material_service.py`, `gui_card_dialogs.py`

Move the six-step DB write sequence from `SaveToCardDialog._on_save`
(lines 380–538) into `material_service.save_to_card()`.

The six steps that move:
1. Get or create print config
2. Build and save microstructure snapshot (dedup check included)
3. Save inference run
4. Save constituent properties (free → inferred, fixed → inputted with dedup guard)
5. Save composite properties
6. Save experimental measurements (conditional on `save_experimental` flag)

After extraction, `_on_save` becomes:
```python
def _on_save(self):
    fid = self._fiber_map.get(self._fiber_var.get())
    pid = self._polymer_map.get(self._polymer_var.get())
    if fid is None or pid is None:
        messagebox.showerror(...)
        return
    card_val = self._card_var.get()
    card_id  = None if card_val == "new" else int(card_val)
    try:
        saved_id = material_service.save_to_card(
            result=self.result,
            fiber_id=fid,
            polymer_id=pid,
            card_id=card_id,
            card_name=self._card_name_var.get().strip(),
            printer_id=self._printer_map.get(self._printer_var.get()),
            save_experimental=self._save_exp_var.get(),
            notes=self._notes_var.get().strip(),
        )
    except Exception as exc:
        messagebox.showerror("Save Error", str(exc), parent=self._win)
        return
    messagebox.showinfo("Saved", f"Saved to card id={saved_id}", parent=self._win)
    self._win.destroy()
```

**Verification:** EVALUATION_PLAN.md Section 6 — all DB integrity checks pass.
`_MICRO_DB_MAP`, `_FIBER_FIELDS`, `_POLYMER_FIELDS` field classification sets stay
in `gui_card_dialogs.py` for now (they are still needed to build the summary label).

---

### Phase 4 — Extract `run_inverse()`

**Files changed:** `material_service.py`, `gui_inverse.py`

Move the solver body from `_solve_worker` (lines 691–761) into
`material_service.run_inverse()`. This includes:
- Building the `InverseProblem` from `fixed_inputs` + `free_fields`
- Running the optimiser
- Assembling the `InverseResult`

What **stays** in `gui_inverse.py`:
- The threading wrapper (`threading.Thread(target=...).start()`)
- The `root.after(0, ...)` callback that posts the result back to the GUI thread
- The `_on_solve_ok()` display logic (writing text to the result panel)
- `_last_result` storage (still the same `InverseResult` dataclass)

After extraction, `_solve_worker` becomes:
```python
def _solve_worker(self, fixed_inputs, free_fields, bounds, init_free,
                  target_outputs, sigmas, solver_cfg, tags):
    try:
        result = material_service.run_inverse(
            fiber_id=self._card_fiber_id,
            polymer_id=self._card_polymer_id,
            fixed_inputs=fixed_inputs,
            free_fields=free_fields,
            bounds=bounds,
            target_outputs=target_outputs,
            sigmas=sigmas,
            model=self._model_var.get(),
            solver_cfg=solver_cfg,
        )
        result_with_tags = {**dataclasses.asdict(result), "tags": tags}
        self.root.after(0, lambda r=result_with_tags: self._on_solve_ok(r))
    except Exception as exc:
        self.root.after(0, lambda e=exc: self._on_solve_error(str(e)))
```

**Verification:** EVALUATION_PLAN.md Section 3.3 (elastic solve), 3.7
(thermoelastic solve). Loss values should be identical to pre-refactor.

---

### Phase 5 — Extract `predict()` for forward GUI

**Files changed:** `material_service.py`, `gui.py`

Move the forward prediction worker in `gui.py` into `material_service.predict()`.

`predict()` takes `fiber_id`, `polymer_id`, a flat `microstructure` dict, and
`model` name. It calls `db.fiber_model_inputs()` + `db.polymer_model_inputs()`,
merges with the microstructure dict, calls `forward.load_forward(model).predict()`,
and returns a `PredictionResult`.

**Verification:** EVALUATION_PLAN.md Section 2.7 (forward prediction), 2.9
(thermal forward). Output values must match pre-refactor.

---

### Phase 6 — Extract `run_thermal_inverse()` and `save_thermal_to_card()`

**Files changed:** `material_service.py`, `gui_thermal_inverse.py`,
`gui_card_dialogs.py`

Move the thermal solver body from `gui_thermal_inverse.py` into
`material_service.run_thermal_inverse()`, returning a `ThermalInverseResult`.

Move `SaveThermalToCardDialog._on_save` DB writes into
`material_service.save_thermal_to_card()` (this is a thin wrapper over the
existing `db.save_thermal_inverse_results()`).

**Verification:** EVALUATION_PLAN.md Section 4.3 (thermal estimation), Section 7
(thermal helper functions).

---

### Phase 7 — Final verification: confirm zero tkinter dependency in service

After all phases are complete, run:

```bash
python -c "
import sys
import material_service
mods = [m for m in sys.modules if 'tkinter' in m]
assert not mods, f'tkinter leaked into service layer: {mods}'
print('Service layer is GUI-free. OK')
"
```

This is the contract. If it passes, any frontend can call `material_service`
without having a display or a Tk root.

Also run a smoke test that exercises the full service without any GUI:

```python
# smoke_test_service.py
import material_service as svc

fibers   = svc.list_fibers()
polymers = svc.list_polymers()
assert fibers and polymers

fid = fibers[0]["id"]
pid = polymers[0]["id"]

# Forward prediction
result = svc.predict(fid, pid, {
    "a11": 0.6, "a22": 0.2, "a12": 0.0, "a13": 0.0, "a23": 0.0,
    "ar": 20.0, "fiber_massfrac": 0.3,
}, model="elastic")
assert "E1" in result.outputs
print("predict() OK — E1 =", result.outputs["E1"])

# Inverse solve (small problem, fast)
inv = svc.run_inverse(
    fiber_id=fid,
    polymer_id=pid,
    fixed_inputs={k: v for k, v in result.inputs.items()
                  if k not in ("matrix_modulus",)},
    free_fields=["matrix_modulus"],
    bounds={"matrix_modulus": (2000.0, 6000.0)},
    target_outputs={"E1": result.outputs["E1"]},
    sigmas={},
    model="elastic",
)
print("run_inverse() OK — loss =", inv.loss, "  matrix_modulus =",
      inv.free_variables.get("matrix_modulus"))

# Load card (requires at least one card in DB)
cards = svc.list_cards()  # add this helper in Phase 2
if cards:
    inputs = svc.load_card_inputs(cards[0]["id"])
    print("load_card_inputs() OK — provenance keys:", list(inputs.provenance.keys()))

print("All service smoke tests passed.")
```

Run with: `python smoke_test_service.py`

---

## File inventory after all phases

| File | Status | Notes |
|---|---|---|
| `service_types.py` | **New** | Four dataclasses; no imports from db/gui |
| `material_service.py` | **New** | All business logic; no tkinter import |
| `db.py` | Unchanged | Raw CRUD only |
| `forward.py` | Unchanged | Pure computation |
| `inverse.py` | Unchanged | Pure computation |
| `inverse_thermal.py` | Unchanged | Pure computation |
| `gui.py` | Thinner | Predict callback delegates to `svc.predict()` |
| `gui_inverse.py` | Thinner | `_solve_worker` delegates to `svc.run_inverse()` |
| `gui_thermal_inverse.py` | Thinner | Solve worker delegates to `svc.run_thermal_inverse()` |
| `gui_card_dialogs.py` | Thinner | `_on_save` / `_on_load` delegate to service; field classification sets stay |

---

## What becomes possible after this refactor

| Use case | What you build | Calls |
|---|---|---|
| Virtual engineer / AI agent | Python script or agent loop | `material_service` directly |
| REST API | FastAPI with async job queue | `material_service` in worker thread |
| Web UI | React + FastAPI | Same FastAPI endpoints above |
| Streamlit prototype | Single `.py` file | `material_service` directly |
| Multi-user cloud | Add `user_id` param to `save_to_card()` + schema migration | No other changes needed |
| Unit tests for solver logic | `pytest` with mock DB | `material_service.run_inverse()` directly |

---

## Out of scope for this refactor

- Schema changes (user isolation, Postgres migration)
- Replacing Tkinter with another GUI framework
- Job queue / async execution for cloud deployment
- Model checkpoint management / lazy loading
- Any changes to the surrogate training code

Those are follow-on work. This plan only creates the stable boundary that makes
them independently tractable.
