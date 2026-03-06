# Surrogate Deployment – Instructions

## Folder structure

```
final/
├── INSTRUCTIONS.md          ← this file
├── gui.py                   ← forward evaluation GUI (run this)
├── inverse.py               ← inverse design solver (run this)
├── forward.py               ← Python API (import this in scripts)
├── problem.json             ← inverse problem definition (edit this)
└── models/
    ├── elastic/
    │   ├── model_config.json        ← arch + field names
    │   ├── normalization_stats.npz  ← drop in from EL_surrogate/
    │   └── ckpt/
    │       └── <checkpoint_name>/   ← drop in from EL_surrogate/ckpt/
    └── thermoelastic/
        ├── model_config.json
        ├── normalization_stats.npz
        └── ckpt/
            └── <checkpoint_name>/
```

`model_config.json` is the only file that needs to be manually maintained (or
auto-generated via the export scripts). It tells `forward.py` which architecture
to reconstruct and which field names correspond to each input/output position.


---

## Step 1 – Export artifacts after training

### Elastic surrogate
```bash
cd EL_surrogate
python export_model.py --config configs/case_5.py --workdir .
```
This writes into `final/models/elastic/`:
- `model_config.json`
- `normalization_stats.npz`
- `ckpt/case_5/`

Change `--config` if you trained a different case. The `--out` flag overrides
the destination if needed.

### Thermoelastic surrogate
```bash
cd THEL_surrogate
python export_model.py --config configs/case_4.py --workdir .
```
Writes into `final/models/thermoelastic/`.


### Manual drop-in (without the export script)
Copy the three items below into `models/<elastic|thermoelastic>/`:

| What to copy | From |
|---|---|
| `normalization_stats.npz` | `<training_folder>/normalization_stats.npz` |
| `ckpt/<name>/` | `<training_folder>/ckpt/<name>/` |
| `model_config.json` | fill in manually – see below |

`model_config.json` fields:

| Key | What it is | Example |
|---|---|---|
| `arch_name` | Architecture class (`"Mlp"`) | `"Mlp"` |
| `hidden_dim` | Hidden layer sizes (list of ints) | `[128, 512, 256, 256, 128, 64, 16]` |
| `activation` | Activation function string | `"swish"` |
| `use_l2reg` | Whether L2 regularisation was used | `true` |
| `checkpoint_name` | `wandb.name` from the training config | `"case_5"` |
| `input_fields` | Ordered list of input feature names | see below |
| `output_fields` | Ordered list of output feature names | see below |

**Field names must exactly match the order used during training.**

Elastic input fields (16):
```json
["e1","e2","g12","f_nu12","f_nu23","ar","fiber_massfrac","fiber_density",
 "matrix_modulus","matrix_poisson","matrix_density","a11","a22","a12","a13","a23"]
```
Elastic output fields (9):
```json
["E1","E2","E3","G12","G13","G23","nu12","nu13","nu23"]
```

Thermoelastic input fields (19):
```json
["e1","e2","g12","f_nu12","f_nu23","f_cte1","f_cte2","ar","fiber_massfrac",
 "fiber_density","matrix_modulus","matrix_poisson","matrix_density","m_cte",
 "a11","a22","a12","a13","a23"]
```
Thermoelastic output fields (15):
```json
["E1","E2","E3","G12","G13","G23","nu12","nu13","nu23",
 "CTE11","CTE22","CTE33","CTE12","CTE13","CTE23"]
```


---

## Step 2 – Forward evaluation GUI

```bash
cd final
python gui.py
```

1. Select **Elastic** or **Thermoelastic** using the radio buttons.
2. Click **Load Model** – the input and output panels populate automatically.
   A green status message confirms a successful load.
3. Fill in the input fields (all fields are required).
   Press **Enter** in any field or click **Predict**.
4. Predicted outputs appear on the right (in GPa / µ/K).
5. The history plot at the bottom tracks successive predictions.
   Click **Clear History** to reset the plot.


---

## Step 3 – Inverse design (elastic surrogate only)

Edit `problem.json` to define your problem, then run:

```bash
cd final
python inverse.py
# or with a specific file:
python inverse.py --problem my_case.json
```

Results are printed to the terminal and saved as `<problem_stem>_result.json`.

### problem.json schema

```jsonc
{
  // All model inputs with their values.
  // Free variables still need a value here — it is used as the initial guess.
  "fixed_inputs": {
    "e1": 240000.0,
    "matrix_modulus": 3100.0,
    "a11": 0.6,
    ...
  },

  // Which inputs the solver is allowed to change.
  "free_inputs": ["matrix_modulus", "matrix_poisson", "a11", "a22"],

  // Optional per-variable box constraints [lo, hi].
  "bounds": {
    "matrix_modulus": [2000.0, 5000.0],
    "a11": [0.2, 0.81]
  },

  // Surrogate outputs to match.
  "target_outputs": {
    "E1": 15420.0,
    "E2": 5140.0
  },

  // Solver settings.
  "solver": {
    "method": "lbfgs",      // "lbfgs" | "lbfgsb" | "adam"
    "maxiter": 300,
    "tol": 1e-6,
    "constraint_penalty": 10000.0,
    // Adam-only options (ignored for lbfgs):
    "lr": 0.01,
    "n_steps": 5000
  },

  // Optional: override the initial guess for free variables.
  // Must be a list in the same order as free_inputs.
  "init_free": [3100.0, 0.37, 0.6, 0.1]
}
```

**Notes:**
- `fixed_inputs` must include *all* model inputs (all 16 for elastic).
  Values for free variables act as the initial guess unless `init_free` is set.
- The `a11 + a22 <= 1.0` constraint is added automatically when both are free.
- All moduli are in **MPa**; all CTE values are in **1/K**.


---

## Python API (for scripting or cloud use)

```python
from forward import load_forward

fwd, meta = load_forward("elastic")        # or "thermoelastic"

outputs = fwd({
    "e1": 240e3, "e2": 15e3, "g12": 28e3, "f_nu12": 0.2, "f_nu23": 0.4,
    "ar": 20.0, "fiber_massfrac": 0.20, "fiber_density": 1780.0,
    "matrix_modulus": 3100.0, "matrix_poisson": 0.37, "matrix_density": 1280.0,
    "a11": 0.6, "a22": 0.1, "a12": 0.0, "a13": 0.0, "a23": 0.0,
})
print(outputs["E1"])   # MPa

# Available metadata
print(meta.input_fields)   # ordered list of input names
print(meta.output_fields)  # ordered list of output names
```

For cloud deployment, copy `NN_surrogate/` from `EL_surrogate/` into `final/`
and remove the `sys.path` manipulation in `forward.py`.
