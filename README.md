# DiffMicromechanics – Composite Material Surrogate Models

This repository contains neural network surrogate models for predicting the effective mechanical and thermoelastic properties of short-fiber composite materials. The trained, ready-to-use deployment lives in the `final/` folder.

> **New here? Read the setup guide first:**
> **[SETUP_AND_RUN.md](SETUP_AND_RUN.md)** – step-by-step instructions for
> installing Python, creating an environment, installing all dependencies
> (Windows / Mac / Linux, CPU and GPU), and running both GUIs.

---

## Repository Structure

```
DiffMicromechanics/
├── SETUP_AND_RUN.md         <- Start here – full setup & GUI run guide
├── final/                   <- Deployed surrogates (run the GUIs from here)
├── EL_surrogate/            <- Elastic surrogate training code
├── THEL_surrogate/          <- Thermoelastic surrogate training code
├── TC_surrogate/            <- Thermal conductivity surrogate training code
├── surrogate/               <- Shared training utilities, model definitions, configs
├── diffmicro/               <- Core micromechanics library
├── main.py                  <- Top-level training entry point
└── README.md
```

---

## Quickstart – `final/` Folder

The `final/` folder contains everything needed to run forward predictions and inverse design without touching the training code.

```
final/
├── gui.py               <- Forward evaluation GUI (Tkinter)
├── gui_inverse.py       <- Inverse design GUI
├── forward.py           <- Python API for scripting
├── inverse.py           <- Inverse design solver (CLI)
├── problem.json         <- Inverse problem definition (edit this)
├── field_labels.json    <- Human-readable field name mappings
└── models/
    ├── elastic/         <- Elastic surrogate (9 outputs)
    │   ├── model_config.json
    │   ├── normalization_stats.npz
    │   └── ckpt/case_5/
    └── thermoelastic/   <- Thermoelastic surrogate (15 outputs)
        ├── model_config.json
        ├── normalization_stats.npz
        └── ckpt/case_4/
```

### Models

| Model | Inputs | Outputs |
|---|---|---|
| `elastic` | 16 fiber/matrix/orientation parameters | E1, E2, E3, G12, G13, G23, nu12, nu13, nu23 |
| `thermoelastic` | 19 parameters (elastic + CTE inputs) | 9 elastic + CTE11, CTE22, CTE33, CTE12, CTE13, CTE23 |

All moduli are in **MPa**. CTE values are in **1/K**.

---

## Running the Forward Model

### Option 1 – GUI

```bash
cd final
python gui.py
```

1. Select **Elastic** or **Thermoelastic** using the radio buttons.
2. Click **Load Model** – input and output panels populate automatically.
3. Fill in all input fields and press **Enter** or click **Predict**.
4. Predicted outputs appear on the right. A history plot tracks successive predictions.

### Option 2 – Python API

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

print(meta.input_fields)   # ordered list of input names
print(meta.output_fields)  # ordered list of output names
```

Run from the `final/` directory so that `forward.py` can resolve the model paths and the shared `EL_surrogate/` code.

---

## Running the Inverse Design Solver

The inverse solver finds input parameters that produce a desired set of output properties. It uses the elastic surrogate by default.

### Step 1 – Edit `problem.json`

```jsonc
{
  // All 16 model inputs. Values for free variables serve as the initial guess.
  "fixed_inputs": {
    "e1": 240000.0,
    "matrix_modulus": 3100.0,
    ...
  },

  // Which inputs the solver is allowed to change.
  "free_inputs": ["matrix_modulus", "matrix_poisson", "a11", "a22"],

  // Optional per-variable box constraints [lo, hi].
  "bounds": {
    "matrix_modulus": [2000.0, 5000.0],
    "a11": [0.2, 0.81]
  },

  // Target surrogate outputs to match.
  "target_outputs": {
    "E1": 15420.0,
    "E2": 5140.0
  },

  // Solver settings.
  "solver": {
    "method": "lbfgs",      // "lbfgs" | "lbfgsb" | "adam"
    "maxiter": 300,
    "tol": 1e-6,
    "constraint_penalty": 10000.0
  }
}
```

**Notes:**
- `fixed_inputs` must include **all** 16 model inputs. Free variable values act as the initial guess.
- The constraint `a11 + a22 <= 1.0` is enforced automatically when both are free.

### Step 2 – Run the solver

```bash
cd final
python inverse.py                           # uses problem.json
python inverse.py --problem my_case.json    # custom problem file
```

Results are printed to the terminal and saved as `<problem_stem>_result.json`.

### Solver methods

| Method | Description |
|---|---|
| `lbfgs` | L-BFGS (default, unconstrained) |
| `lbfgsb` | L-BFGS-B (bounded, uses `bounds` from JSON) |
| `adam` | Adam gradient descent (use `lr` and `n_steps` options) |

---

## Training the Surrogates

Training is driven by configuration files in each surrogate folder.

### Elastic surrogate

```bash
python main.py --config EL_surrogate/configs/case_5.py --workdir EL_surrogate/
```

### Thermoelastic surrogate

```bash
python main.py --config THEL_surrogate/configs/case_4.py --workdir THEL_surrogate/
```

After training, export model artifacts into `final/models/`:

```bash
cd EL_surrogate
python export_model.py --config configs/case_5.py --workdir .

cd THEL_surrogate
python export_model.py --config configs/case_4.py --workdir .
```

See `final/INSTRUCTIONS.md` for the full export and manual drop-in workflow.

---

## Requirements

See `requirements.txt` for the full dependency list. Key packages:

- `jax`, `jaxlib` – core numerical backend
- `flax` – neural network layers
- `optax` – optimisers
- `jaxopt` – L-BFGS / L-BFGS-B solvers for inverse design
- `orbax-checkpoint` – checkpoint save/restore
- `ml-collections` – configuration management
- `numpy`, `scipy`, `matplotlib`, `pandas`
- `wandb` – experiment tracking (training only)

Install with:

```bash
pip install -r requirements.txt
```
