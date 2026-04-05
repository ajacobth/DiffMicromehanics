# Service Layer Summary

How to drive the framework programmatically — and how the GUI uses the same functions under the hood.

---

## 1. How a GUI starts and stays alive

Every GUI file follows the same three-step pattern:

```
python gui.py
    └─ main()
        ├─ root = tk.Tk()           ← creates the OS window
        ├─ SurrogateGUI(root)       ← builds all widgets, wires up button callbacks
        └─ root.mainloop()          ← hands control to Tkinter — blocks here forever
```

`root.mainloop()` is the event loop. It does nothing but wait. When you click a button, Tkinter fires the Python callback you registered. When the callback returns, `mainloop` goes back to waiting. The Python process never exits until the window is closed.

### Why `root.after(0, callback)` is used inside threads

Long computations (model loading, solving, running the thermal inverse) are run on a **background thread** so the UI doesn't freeze. But Tkinter widgets can only be safely touched from the main thread. The pattern throughout the code is:

```python
def _solve_worker(self, ...):          # runs on background thread
    result = run_inverse(...)          # heavy computation, can take seconds
    self.root.after(0, lambda: self._on_solve_ok(result))
    #  ↑ schedules _on_solve_ok to run on the main thread at the next event-loop tick
```

`root.after(delay_ms, fn)` is Tkinter's thread-safe bridge. Delay `0` means "as soon as possible".

### Each GUI window type

| File | Class | Lives as | Opened by |
|---|---|---|---|
| `gui.py` | `SurrogateGUI` | `tk.Tk` root | `python gui.py` |
| `gui_inverse.py` | `InverseGUI` | `tk.Tk` root | `python gui_inverse.py` |
| `gui_thermal_inverse.py` | `ThermalInverseWindow` | `tk.Toplevel` | `gui_inverse.py` button **or** `python gui_thermal_inverse.py` |
| `gui_identifiability.py` | `IdentifiabilityWindow` | `tk.Toplevel` | `gui.py` button **or** `python gui_identifiability.py` |

`tk.Toplevel` is a child window — it shares the event loop with its parent `tk.Tk`. When opened standalone, a hidden root `tk.Tk` is created just to own the event loop, and the real window is a Toplevel on top of it.

### How child windows stay alive

```python
# gui_inverse.py opens thermal window:
def _open_thermal_window(self):
    win = ThermalInverseWindow(parent=self.root)
    # win is a Toplevel — it is now alive inside root's event loop
    # no extra mainloop() needed — root.mainloop() already handles it
```

A `Toplevel` lives until its OS window is closed or `win.destroy()` is called. Modal dialogs (`grab_set()` + `wait_window()`) block interaction with the parent until they close.

---

## 2. The service layer — everything the GUI can do, you can call directly

### Forward prediction

**What the GUI does:** Load model → fill in inputs → click Predict → see output values.

```python
from core.services import get_model, warm_up_model, run_forward, get_input_fields, get_output_fields

# Optional: warm up JAX JIT so first prediction is fast (run in a thread if needed)
warm_up_model("elastic")          # "elastic" | "thermoelastic" | "thermal"

# Check what fields the model expects
fields_in  = get_input_fields("elastic")   # ['a11', 'a22', ..., 'w_f', ...]
fields_out = get_output_fields("elastic")  # ['E1', 'E2', 'G12', 'nu12']

# Run a prediction — all values in model units (MPa, W/m·K, 1/K, etc.)
outputs = run_forward("elastic", {
    "a11": 0.8, "a22": 0.1, "a12": 0.0, "a13": 0.0, "a23": 0.0,
    "ar_f": 20.0, "w_f": 0.35, "fiber_density": 1800.0, ...
})
# returns: {"E1": 45000.0, "E2": 8000.0, "G12": 3200.0, "nu12": 0.34}

# If you want a cached model object (to inspect input/output field lists, indices, etc.)
model = get_model("elastic")
print(model.input_fields)   # list of input field names in model order
print(model.output_fields)  # list of output field names in model order
```

---

### Elastic / thermoelastic inverse

**What the GUI does:** Mark some inputs as Free, enter target outputs, click Solve → get optimised free-variable values.

```python
from core.services import run_inverse, validate_orientation_tensor

# (Optional) check the orientation tensor is physically valid before solving
err = validate_orientation_tensor({"a11": 0.8, "a22": 0.1, "a12": 0.0, "a13": 0.0, "a23": 0.0})
if err:
    raise ValueError(err)

result = run_inverse(
    model_name     = "elastic",          # "elastic" | "thermoelastic"
    fixed_inputs   = {
        "a11": 0.8, "a22": 0.1, "a12": 0.0, "a13": 0.0, "a23": 0.0,
        "ar_f": 20.0,
        # fiber stiffness from datasheet:
        "e1": 230000.0, "e2": 15000.0, "g12": 15000.0, "f_nu12": 0.2, "f_nu23": 0.25,
    },
    free_inputs    = ["matrix_modulus", "matrix_poisson", "w_f"],
    bounds         = {
        "matrix_modulus": (1000.0, 8000.0),
        "matrix_poisson": (0.3, 0.45),
        "w_f":            (0.2, 0.5),
    },
    target_outputs = {"E1": 45000.0, "E2": 9000.0, "G12": 3500.0, "nu12": 0.32},
    sigmas         = {"E1": 500.0, "E2": 200.0, "G12": 100.0, "nu12": 0.01},
    init_vals      = [3500.0, 0.37, 0.35],   # optional: your best initial guess
    solver_cfg     = {"method": "lbfgsb", "maxiter": 300, "tol": 1e-9},
)

# result is an InverseResult TypedDict:
result["opt_free"]          # {"matrix_modulus": 3820.0, "matrix_poisson": 0.37, "w_f": 0.33}
result["predicted_outputs"] # {"E1": 44950.0, "E2": 9020.0, "G12": 3490.0, "nu12": 0.319}
result["target_outputs"]    # your target dict (echoed back)
result["final_error"]       # 1.23e-5  (optimiser loss)
result["model"]             # "elastic"
result["solver_cfg"]        # the cfg used
```

**Thermoelastic** works identically — switch `model_name="thermoelastic"` and free `f_cte1`, `f_cte2`, `m_cte` instead. Fixed inputs must include all microstructure + matrix modulus (loaded from a card after elastic inverse).

---

### Thermal inverse

**What the GUI does:** Pick a CSV of K vs T measurements, set microstructure inputs, click Run → recover p1, p2, l2, t.

```python
from core.services import load_thermal_data, vf_to_wf, run_thermal_inverse, compute_conductivity_curves
import numpy as np

# 1. Convert volume fraction → mass fraction if needed
w_f = vf_to_wf(vf=0.35, rho_f=1800.0, rho_m=1260.0)

# 2. Load measurement data from CSV or Excel
#    CSV must have columns: temperature, K11 (and optionally K22, K33)
temperatures, K_data = load_thermal_data("data/thermal_measurements.csv")
# temperatures: np.ndarray shape (N,)
# K_data: {"K11": np.ndarray|None, "K22": np.ndarray|None, "K33": np.ndarray|None}

# 3. Run the inverse
def on_progress(restart_i, n_restarts, loss):
    print(f"Restart {restart_i}/{n_restarts}  loss={loss:.4e}")

result = run_thermal_inverse(
    fixed_inputs = {
        "ar_f": 20.0, "w_f": w_f,
        "rho_f": 1800.0, "rho_m": 1260.0,
        "a11": 0.8, "a22": 0.1, "a12": 0.0, "a13": 0.0, "a23": 0.0,
    },
    temperatures = temperatures,
    K_data       = K_data,
    n_restarts   = 10,
    seed         = 42,
    progress_cb  = on_progress,   # optional
)

# result is a ThermalResult TypedDict:
result["p1"]           # polymer conductivity scaling coefficient
result["p2"]           # polymer conductivity offset
result["l2"]           # fiber longitudinal conductivity (= k_f1)
result["t"]            # fiber anisotropy ratio (k_f1 / k_f2)
result["best_loss"]    # final MSE
result["temperatures"] # list[float] — same as input
result["K_pred"]       # {"K11": list[float], "K22": list[float], "K33": list[float]}

# 4. Compute smooth constituent conductivity curves (e.g. for plotting)
T_range = np.linspace(20, 120, 200)
curves = compute_conductivity_curves(result["p1"], result["p2"],
                                     result["l2"], result["t"], T_range)
curves["k_polymer"]     # list[float] — K_m(T)
curves["k_fiber_long"]  # list[float] — constant = l2
curves["k_fiber_trans"] # list[float] — constant = l2/t
```

---

### Identifiability analysis (FIM)

**What the GUI does:** Choose free variables + target outputs → check if the experiment can distinguish each parameter.

```python
from core.services import run_fim

result = run_fim(
    model_name     = "elastic",
    fixed_inputs   = {"a11": 0.8, "a22": 0.1, "a12": 0.0, "a13": 0.0, "a23": 0.0,
                      "ar_f": 20.0, "e1": 230000.0, ...},
    free_inputs    = ["matrix_modulus", "w_f"],
    bounds         = {"matrix_modulus": (1000.0, 8000.0), "w_f": (0.2, 0.5)},
    target_outputs = {"E1": 0.0, "E2": 0.0},   # values don't matter for identifiability
    sigmas         = {"E1": 500.0, "E2": 200.0},
    n_samples      = 200,
)

result["status"]           # {"matrix_modulus": "WELL", "w_f": "MARGINAL"}
result["cramer_rao_std"]   # {"matrix_modulus": 45.3, "w_f": 0.012}  — uncertainty estimates
result["recommendations"]  # list of suggestion strings
```

---

### Material library

```python
from core.services import (
    list_fibers, list_polymers, list_printers, list_cards,
    get_fiber_inputs, get_polymer_inputs,
    get_inferred_inputs, has_inferred_data,
    get_model_inputs,
    add_fiber, add_polymer, add_printer,
)

# Browse the library
fibers   = list_fibers()    # [{"id": 1, "name": "Carbon Fiber T300", ...}, ...]
polymers = list_polymers()
printers = list_printers()
cards    = list_cards()     # all (fiber, polymer, printer) triples

# Retrieve datasheet values for a material
fiber_inputs = get_fiber_inputs(fiber_id=1)
# {"e1": 230000.0, "e2": 15000.0, "g12": 15000.0, "f_nu12": 0.2, ...}

polymer_inputs = get_polymer_inputs(polymer_id=1)
# {"matrix_modulus": 3800.0, "matrix_poisson": 0.4, ...}

# Check if inferred properties exist for a pair
if has_inferred_data(fiber_id=1, polymer_id=1):
    inferred = get_inferred_inputs(fiber_id=1, polymer_id=1)
    # {"matrix_modulus": 3820.0, "f_cte1": 0.0, ...}  — from previous inverse run

# Get combined inputs — datasheet + inferred merged
inputs = get_model_inputs(fiber_id=1, polymer_id=1, use_inferred=True)

# Add a new material
add_fiber(name="AS4", supplier="Hexcel", e1=228000.0, ...)
add_polymer(name="PEEK", supplier="Victrex", matrix_modulus=4200.0, ...)
add_printer(name="Markforged X7", supplier="Markforged")
```

---

### Material cards (save / load)

**What the GUI does after Solve:** Click "Save to Card" → store microstructure + inferred properties against a (fiber, polymer, printer) triple.

```python
from core.services import (
    load_card, load_card_inputs,
    save_inverse_result, save_forward_result,
    save_thermal_result,
)

# Save an elastic/thermoelastic inverse result to card
save_inverse_result(
    result         = result,        # InverseResult from run_inverse()
    fiber_id       = 1,
    polymer_id     = 1,
    printer_id     = 2,
    card_name      = "T300/PESU on Markforged X7",
)

# Save a forward prediction to card
save_forward_result(
    model_name     = "elastic",
    inputs         = {...},
    outputs        = {...},
    fiber_id       = 1,
    polymer_id     = 1,
    printer_id     = 2,
)

# Save thermal inverse result to card
save_thermal_result(
    result     = thermal_result,    # ThermalResult from run_thermal_inverse()
    fiber_id   = 1,
    polymer_id = 1,
    printer_id = 2,
)

# Load a card by its ID
card = load_card(print_config_id=3)
# {"fiber": {...}, "polymer": {...}, "printer": {...}, "microstructure": {...}, ...}

# Load a card as a flat input dict (ready to pass to run_inverse / run_forward)
inputs = load_card_inputs(print_config_id=3)
# {"a11": 0.8, "a22": 0.1, "w_f": 0.33, "matrix_modulus": 3820.0, ...}
```

---

### Transfer to a new printer (Stage 4 workflow)

```python
from core.services import resolve_constituent_props, build_forward_inputs, run_transfer

# See what constituent properties have been inferred for a card
props = resolve_constituent_props(card_id=3)
# {"matrix_modulus": {"value": 3820.0, "source_tag": "inferred"}, "k_f1": {...}, ...}

# Build complete forward inputs from a card + new microstructure
inputs = build_forward_inputs(
    card_id       = 3,
    microstructure = {"a11": 0.7, "a22": 0.15, "a12": 0.0, "a13": 0.0, "a23": 0.0,
                      "ar_f": 20.0, "w_f": 0.35},
)

# Full transfer: infer on source card, then predict on new microstructure
result = run_transfer(
    source_card_id = 3,
    target_outputs = {"E1": 50000.0},
    ...
)
```

---

## 3. Agentic loop example

With the service layer cleanly separated, an agent can reproduce the full four-stage workflow without any GUI:

```python
from core.services import (
    list_fibers, list_polymers, add_printer,
    run_inverse, save_inverse_result,
    run_thermal_inverse, load_thermal_data, vf_to_wf, save_thermal_result,
    load_card_inputs, run_forward, save_forward_result,
)

# Stage 0: set up
fibers   = list_fibers()
polymers = list_polymers()
fid, pid = fibers[0]["id"], polymers[0]["id"]
add_printer(name="Agent Printer", supplier="Test")

# Stage 1: elastic inverse
r1 = run_inverse(
    model_name="elastic",
    fixed_inputs=get_fiber_inputs(fid),
    free_inputs=["matrix_modulus", "matrix_poisson", "w_f", "a11", "a22", ...],
    bounds={...}, target_outputs={"E1": 45000.0, ...},
)
save_inverse_result(r1, fiber_id=fid, polymer_id=pid, printer_id=1, card_name="Card A")

# Stage 2: thermoelastic inverse (loads card to get fixed microstructure + E_m)
base = load_card_inputs(print_config_id=1)
r2 = run_inverse(
    model_name="thermoelastic",
    fixed_inputs=base,
    free_inputs=["f_cte1", "f_cte2", "m_cte"],
    bounds={...}, target_outputs={"CTE11": 2e-6, "CTE22": 30e-6},
)
save_inverse_result(r2, fiber_id=fid, polymer_id=pid, printer_id=1)

# Stage 3: thermal inverse
temperatures, K_data = load_thermal_data("data/thermal.csv")
r3 = run_thermal_inverse(
    fixed_inputs={**base, "rho_f": 1800.0, "rho_m": 1260.0},
    temperatures=temperatures, K_data=K_data, n_restarts=10,
)
save_thermal_result(r3, fiber_id=fid, polymer_id=pid, printer_id=1)

# Stage 4: forward predict on new printer
new_inputs = load_card_inputs(print_config_id=1)
new_inputs.update({"a11": 0.5, "a22": 0.25})   # different orientation
outputs = run_forward("elastic", new_inputs)
print(outputs)   # {"E1": ..., "E2": ..., ...}
```

No windows opened. No threads. No callbacks. Pure function calls.
