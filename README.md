# DiffMicromechanics – Composite Material Surrogate Models

This repository contains neural network surrogate models for predicting the effective mechanical, thermoelastic, and thermal conductivity properties of short-fiber composite materials. The trained, ready-to-use deployment lives in the `final/` folder.

> **New here? Read the setup guide first:**
> **[SETUP_AND_RUN.md](SETUP_AND_RUN.md)** – step-by-step instructions for
> installing Python, creating an environment, installing all dependencies
> (Windows / Mac / Linux, CPU and GPU), and running all GUIs.

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
├── gui.py                    <- Forward evaluation GUI (elastic, thermoelastic, thermal)
├── gui_inverse.py            <- Elastic / thermoelastic inverse design GUI
├── gui_thermal_inverse.py    <- Thermal conductivity inverse estimation GUI
├── gui_material_card.py      <- Material card viewer
├── gui_card_dialogs.py       <- Save/Load card dialogs
├── gui_identifiability.py    <- Parameter identifiability analysis GUI
│
├── core/                     <- Computation / solver logic
│   ├── forward.py            <- Python API for scripting forward predictions
│   ├── inverse.py            <- Elastic / thermoelastic inverse solver
│   ├── inverse_thermal.py    <- Thermal inverse solver core
│   ├── fim.py                <- Fisher Information Matrix utilities
│   ├── micro_surrogate.py    <- Surrogate model class definitions
│   └── unit_manager.py       <- Unit conversion utilities
│
├── db/                       <- Database layer
│   ├── db.py                 <- All DB helper functions
│   └── init_db.py            <- Run once to create/seed the database
│
├── config/                   <- Problem definitions + field labels
│   ├── problem.json          <- Inverse problem definition (edit this)
│   ├── thermal_problem.json  <- Thermal inverse CLI problem file
│   └── field_labels.json     <- Human-readable field name mappings
│
├── scripts/                  <- CLI utilities
│   ├── run_inverse_thermal.py  <- Thermal inverse CLI (no GUI)
│   └── test_setup.py           <- Run this to verify your environment
│
├── NN_surrogate/             <- Neural network base architecture (do not modify)
└── models/
    ├── elastic/              <- Elastic surrogate (9 outputs)
    │   ├── model_config.json
    │   ├── normalization_stats.npz
    │   └── ckpt/case_5/
    ├── thermoelastic/        <- Thermoelastic surrogate (15 outputs)
    │   ├── model_config.json
    │   ├── normalization_stats.npz
    │   └── ckpt/case_4/
    └── thermal/              <- Thermal conductivity surrogate (6 outputs)
        ├── model_config.json
        ├── normalization_stats.npz
        └── ckpt/case_6/
```

### Models

| Model | Inputs | Outputs |
|---|---|---|
| `elastic` | 16 fiber/matrix/orientation parameters | E1, E2, E3, G12, G13, G23, nu12, nu13, nu23 |
| `thermoelastic` | 19 parameters (elastic + CTE inputs) | 9 elastic + CTE11, CTE22, CTE33, CTE12, CTE13, CTE23 |
| `thermal` | 12 parameters (fiber/matrix conductivities, morphology) | k11, k12, k13, k22, k23, k33 |

Elastic/shear moduli are in **MPa**. CTE values are in **1/K**. Thermal conductivities are in **W/(m·K)**.

---

## Running the Forward Model

### Option 1 – GUI

```bash
cd final
python gui.py
```

1. Select **Elastic**, **Thermoelastic**, or **Thermal** using the radio buttons.
2. Click **Load Model** – input and output panels populate automatically.
3. Fill in all input fields and press **Enter** or click **Predict**.
4. Predicted outputs appear on the right. A history plot tracks successive predictions.

**Thermal model inputs** (12 fields):

| Input | Description | Units |
|---|---|---|
| `k_f1` | Fiber longitudinal thermal conductivity | W/(m·K) |
| `k_f2` | Fiber transverse thermal conductivity | W/(m·K) |
| `k_m` | Matrix thermal conductivity | W/(m·K) |
| `ar_f` | Fiber aspect ratio | – |
| `w_f` | Fiber mass fraction | – |
| `rho_f` | Fiber density | kg/m³ |
| `rho_m` | Matrix density | kg/m³ |
| `a11`, `a22`, `a12`, `a13`, `a23` | Fiber orientation tensor components | – |

**Thermal model outputs**: composite conductivity tensor components k11, k12, k13, k22, k23, k33 in **W/(m·K)**.

### Option 2 – Python API

```python
from core.forward import load_forward

# Elastic
fwd, meta = load_forward("elastic")        # or "thermoelastic" or "thermal"

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

Run from the `final/` directory so that `forward.py` can resolve the model paths.

---

## Running the Inverse Design Solver

### Elastic / Thermoelastic Inverse

#### Option A – GUI

```bash
cd final
python gui_inverse.py
```

Select **Elastic** or **Thermoelastic**, set fixed/free inputs, specify target outputs, and click **SOLVE**. Results are displayed in-window and can be exported as JSON.

#### Option B – CLI

Edit `problem.json` to describe the problem:

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

```bash
cd final
python core/inverse.py                                  # uses config/problem.json
python core/inverse.py --problem config/my_case.json    # custom problem file
```

Results are printed to the terminal and saved as `<problem_stem>_result.json`.

### Solver methods

| Method | Description |
|---|---|
| `lbfgs` | L-BFGS (default, unconstrained) |
| `lbfgsb` | L-BFGS-B (bounded, uses `bounds` from JSON) |
| `adam` | Adam gradient descent (use `lr` and `n_steps` options) |

---

## Running the Thermal Inverse Estimation

The thermal inverse recovers constituent conductivity parameters (fiber and matrix) from measured composite conductivities (K11, K22, K33) at multiple temperatures. It uses the trained thermal surrogate.

### Option A – GUI (standalone)

```bash
cd final
python gui_thermal_inverse.py
```

### Option B – via the Inverse GUI

```bash
cd final
python gui_inverse.py
```

Click the **Thermal Inverse** button at the top of the window.

### Option C – CLI

Edit `thermal_problem.json`, then run:

```bash
cd final
python scripts/run_inverse_thermal.py                                       # uses config/thermal_problem.json
python scripts/run_inverse_thermal.py --problem config/my_problem.json
python scripts/run_inverse_thermal.py --problem config/p.json --output_dir results/
```

See [SETUP_AND_RUN.md](SETUP_AND_RUN.md) Section 11 for full details on inputs, outputs, and the JSON problem format.

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

### Thermal conductivity surrogate

```bash
python main.py --config TC_surrogate/configs/case_6.py --workdir TC_surrogate/
```

After training, export model artifacts into `final/models/`:

```bash
cd EL_surrogate
python export_model.py --config configs/case_5.py --workdir .

cd THEL_surrogate
python export_model.py --config configs/case_4.py --workdir .

cd TC_surrogate
python export_model.py --config configs/case_6.py --workdir .
```

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
