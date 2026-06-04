# DiffMicromechanics

A Python desktop application for composite micromechanics — predicting and
inferring material properties of fiber-reinforced composites using trained
surrogate neural network models, and transferring characterisation results
between printers without re-running experiments.

---

## Table of Contents

1. [What This Application Does](#1-what-this-application-does)
2. [Entry Points](#2-entry-points)
3. [Directory Structure](#3-directory-structure)
4. [The Material Card System](#4-the-material-card-system)
5. [Forward Prediction](#5-forward-prediction)
6. [Inverse Design](#6-inverse-design)
7. [Thermal Inverse Estimation](#7-thermal-inverse-estimation)
8. [Physics-Informed Transfer Learning](#8-physics-informed-transfer-learning)
9. [Database Design](#9-database-design)
10. [Seeded Material Library](#10-seeded-material-library)

---

## 1. What This Application Does

Composite materials made with short or continuous fiber reinforcement have
properties that depend on three distinct layers:

- **Constituent properties** — the intrinsic stiffness, thermal expansion, and
  conductivity of the fiber and the polymer matrix. These are fixed by chemistry
  and are independent of how the part is printed.
- **Microstructure** — how the fibers are oriented and how much fiber is
  present. This is determined by the printing process and is printer-specific.
- **Composite properties** — what you measure on the finished part: Young's
  moduli, coefficients of thermal expansion, thermal conductivities. These
  emerge from the interaction of constituent properties and microstructure.

This application automates the three-way relationship between these layers using
surrogate neural networks trained on physics-based micromechanics simulations.
It provides four core capabilities:

| Capability | What you provide | What you get |
|---|---|---|
| **Forward prediction** | Constituent inputs + microstructure | Predicted composite properties |
| **Elastic/thermoelastic inverse** | Measured composite properties | Inferred microstructure + constituent properties |
| **Thermal inverse** | Measured K vs T data | Inferred fiber and matrix conductivities |
| **Physics-informed transfer** | Source card + new printer's measurements | Composite property predictions for the new printer |

The key insight that ties all four together is that **constituent properties are
printer-independent**. Once you characterise a fiber-polymer pair on one
printer, those inferred constituent properties can be combined with any other
printer's microstructure to predict composite properties without running new
material experiments.

---

## 2. Entry Points

All commands are run from the `final/` directory.

```bash
# One-time database setup
python db/init_db.py

# Forward prediction GUI (elastic, thermoelastic, thermal)
python gui.py

# Inverse design GUI (elastic + thermoelastic; opens thermal inverse)
python gui_inverse.py

# Thermal inverse GUI (standalone)
python gui_thermal_inverse.py

# Physics-informed transfer learning GUI
python gui_transfer.py

# Material card viewer (standalone)
python gui_material_card.py

# Thermal inverse from the command line
python scripts/run_inverse_thermal.py

# Verify the environment
python scripts/test_setup.py
```

---

## 2b. Chat Agent (MateriAl)

The application includes a conversational AI agent — **MateriAl** — that lets you
run the full characterisation workflow in plain English, without operating the GUI
or writing any code.

### Running the agent

```bash
# From the final/ directory
streamlit run app_chat.py
```

Opens a browser tab with a chat interface. Type your measurements and the agent
runs the correct solvers, enforces stage order, checks results for physical
plausibility, and saves to a material card.

### Model selection

Edit the `ACTIVE_MODEL` line near the top of `app_chat.py`:

```python
MODELS = {
    "haiku":  ("anthropic", "claude-haiku-4-5-20251001"),
    "sonnet": ("anthropic", "claude-sonnet-4-6"),
    "local":  ("ollama",    "qwen2.5:14b-instruct-q4_K_M"),
}
ACTIVE_MODEL = "local"   # change to "haiku" or "sonnet" for Anthropic models
```

**Local model (default):** requires [Ollama](https://ollama.com) installed and the
model pulled:
```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

**Anthropic models:** create a file `final/.env` containing:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Knowledge base (one-time setup)

The agent can answer questions about composite mechanics theory using a PDF
knowledge base. To build it, place PDF references in `agent/knowledge/` then run:

```bash
python agent/build_kb.py
```

This only needs to be done once, or when new PDFs are added.

### What the agent can do

| Prefix | Mode | Example |
|---|---|---|
| `INVERSE` | Characterisation from measurements | `INVERSE E1=15 GPa, E2=5 GPa, mf=0.20, T300/PESU` |
| `PREDICT` | Forward property prediction | `PREDICT card 3 with a11=0.75, mf=0.25` |
| `SEARCH` | Material lookup / theory | `SEARCH what is the CTE of T300?` |
| *(none)* | Agent decides | Works for most requests |

---

## 3. Directory Structure

```
final/
│
├── gui.py                    ← Forward prediction GUI
├── gui_inverse.py            ← Inverse design GUI (elastic + thermoelastic)
├── gui_thermal_inverse.py    ← Thermal inverse GUI
├── gui_transfer.py           ← Physics-informed transfer learning GUI
├── gui_material_card.py      ← Material card viewer (standalone)
├── gui_card_dialogs.py       ← Save to Card / Load from Card dialogs
├── gui_identifiability.py    ← Identifiability analysis GUI
│
├── core/                     ← Computation and solver logic
│   ├── forward.py            ← Forward model loader
│   ├── inverse.py            ← Elastic / thermoelastic inverse solver
│   ├── inverse_thermal.py    ← Thermal inverse solver
│   ├── fim.py                ← Fisher Information Matrix utilities
│   ├── micro_surrogate.py    ← Surrogate model class extensions
│   ├── unit_manager.py       ← Unit conversion singleton (UM)
│   └── services/
│       ├── service_forward.py   ← Forward prediction service (model cache)
│       ├── service_inverse.py   ← Inverse solve service
│       ├── service_cards.py     ← Save-to-card business logic
│       └── service_transfer.py  ← Transfer workflow logic
│
├── db/                       ← Database layer
│   ├── db.py                 ← All DB CRUD helpers (single access point)
│   └── init_db.py            ← DB creation + seeding from JSON
│
├── config/                   ← Problem definitions and field labels
│   ├── field_labels.json     ← Human-readable labels for all model fields
│   ├── problem.json          ← Elastic / thermoelastic problem config
│   └── thermal_problem.json  ← Thermal inverse problem config (CLI)
│
├── scripts/                  ← CLI utilities and dev tools
│   ├── run_inverse_thermal.py  ← Thermal inverse CLI
│   ├── test_setup.py           ← Environment verification
│   └── check_nu_sensitivity.py ← Developer utility
│
├── models/                   ← Pre-trained surrogate model checkpoints
│   ├── elastic/              ← 16 inputs → 9 outputs
│   ├── thermoelastic/        ← 19 inputs → 15 outputs
│   └── thermal/              ← 12 inputs → 6 outputs
│
├── NN_surrogate/             ← Neural network base architecture (do not modify)
│
└── data/                     ← SQLite database + seed JSON files
    ├── micromechanics.db     ← Created by db/init_db.py
    ├── fibers.json           ← Fiber material library seed data
    └── polymers.json         ← Polymer material library seed data
```

---

## 4. The Material Card System

A **material card** is the central data structure of this application. It is a
persistent record of a **(fiber + polymer + printer)** triple stored in the
database. As you work through the characterisation stages, each result is
attached to the card and becomes available to subsequent stages and other GUIs.

### What a card accumulates

```
Stage 1 — Elastic inverse
  → microstructure snapshot: orientation tensor (a11, a22, a12, a13, a23),
                              aspect ratio, mass fraction
  → constituent properties:  matrix modulus, matrix Poisson's ratio (inferred)

Stage 2 — Thermoelastic inverse
  → constituent properties:  fiber CTE (axial + transverse),
                              matrix CTE (inferred, globally reusable)

Stage 3 — Thermal inverse
  → constituent properties:  k_f1, k_f2, k_m at room temperature
                              (inferred, globally reusable)
  → K vs T curve:            K11, K22, K33 predictions over temperature range

Stage 4 — Forward prediction on any printer
  → composite properties:    predicted E1, E2, G12, nu12, CTE11, k11, …
```

### Why constituent properties are stored globally

Matrix modulus, fiber and matrix CTE, and thermal conductivities are intrinsic
to the fiber-polymer chemistry — they do not change with the printer. The
database stores them with `print_config_id = NULL`, meaning they are associated
with the fiber or polymer record, not with a specific card. When you load a card
for a different printer using the same materials, all inferred constituent
properties auto-fill without needing to be re-estimated.

### Provenance tracking

Every stored value carries a provenance tag:

| Tag | Meaning |
|---|---|
| `web` | Manufacturer datasheet value |
| `inputted` | Value entered manually by the user |
| `inferred` | Recovered by an inverse solver |
| `predicted` | Output of a forward surrogate prediction |

The **In-situ** toggle in `gui.py` filters to only `inferred` values when
filling inputs, distinguishing between the raw datasheet and experimentally
characterised constituent properties.

### Viewing a card

```bash
python gui_material_card.py
```

| Tab | Content |
|---|---|
| Summary | Card name, fiber / polymer / printer, microstructure snapshot, row counts |
| Constituent Properties | Datasheet values vs inferred values with provenance and date |
| Microstructure | Full snapshot history with per-field provenance |
| Composite Properties | All predicted and experimental values with source |
| Inference History | Every solver run — click any row to expand inputs, outputs, and loss |

---

## 5. Forward Prediction

**Entry point:** `python gui.py`

The forward GUI takes microstructure and constituent property inputs and
predicts composite material properties using a trained surrogate neural network.
Three surrogate models are available, selectable at the top of the window.

### Surrogate models

#### Elastic model (16 inputs → 9 outputs)

Predicts the full orthotropic elastic stiffness tensor of the composite.

| Inputs | Outputs |
|---|---|
| Fiber: E1, E2, G12, ν12, ν23, ρ | E1, E2, E3 |
| Matrix: E, ν, ρ | G12, G13, G23 |
| Microstructure: a11, a22, a12, a13, a23, ar, mf | ν12, ν13, ν23 |

#### Thermoelastic model (19 inputs → 15 outputs)

Predicts all 9 elastic properties plus 6 thermal expansion coefficients.
Adds fiber CTE (axial + transverse) and matrix CTE to the input set.

| Additional inputs | Additional outputs |
|---|---|
| f_cte1 (fiber axial CTE, 1/K) | CTE11, CTE22, CTE33 |
| f_cte2 (fiber transverse CTE, 1/K) | CTE12, CTE13, CTE23 |
| m_cte (matrix CTE, 1/K) | |

When saving thermoelastic results to a card, only the CTE outputs are
written to the composite property table. The elastic properties (E1–ν23)
are preserved from the dedicated elastic model run and are not overwritten.

#### Thermal conductivity model (12 inputs → 6 outputs)

Predicts the full thermal conductivity tensor of the composite.

| Inputs | Outputs |
|---|---|
| k_f1 (fiber longitudinal conductivity, W/m·K) | k11, k12, k13 |
| k_f2 (fiber transverse conductivity, W/m·K) | k22, k23, k33 |
| k_m (matrix conductivity, W/m·K) | |
| ar_f, w_f, rho_f, rho_m | |
| a11, a22, a12, a13, a23 | |

Because matrix conductivity is temperature-dependent
(`K_m(T) = p1·√(T/T_ref) + p2`), the thermal model must be evaluated
over a range of temperatures to produce a K vs T curve. This is handled
automatically when running thermal predictions through the transfer GUI or
using the `run_thermal_sweep` service function.

### Material auto-fill

Select a fiber and polymer from the dropdowns. The corresponding datasheet
values populate the input fields automatically. Two fill modes are available:

| Mode | Behaviour |
|---|---|
| **Neat** | Fills from manufacturer datasheet values (`web` tag) |
| **In-situ** | Fills from experimentally inferred values stored in the DB (`inferred` tag); only enabled when the selected fiber-polymer pair has inferred constituent properties |

The **In-situ** mode is what makes property transfer useful: once a
fiber-polymer pair has been characterised through the inverse stages, the
In-situ toggle replaces datasheet values with the inferred constituent
properties for that specific combination.

### Load from Card

Click **Load from Card** to auto-fill all inputs from a previously saved
material card. The dialog loads:
- Microstructure snapshot (orientation tensor, aspect ratio, mass fraction)
- Inferred constituent properties (matrix modulus, CTEs, conductivities)
- Falls back to datasheet values for any property not yet inferred

Thermal conductivity values (k_f1, k_f2, k_m) are labelled `[inferred, at 25 °C]`
in the provenance display to indicate they are room-temperature scalars derived
from the temperature-dependent parametric model.

### Identifiability check

Click **Identifiability Check** to open a sensitivity analysis tool built on
the Fisher Information Matrix (FIM). For the current set of inputs, it shows:

- Which input parameters most influence each output
- The condition number of the FIM (a measure of how well-posed the inverse
  problem is for these inputs)
- Per-output sensitivity rankings

This helps you decide which inputs to hold Fixed and which to make Free
before running an inverse solve.

### Saving predictions

After a successful prediction, click **Save Prediction** to attach the predicted
composite properties to a material card in the database with provenance tag
`predicted`.

---

## 6. Inverse Design

**Entry point:** `python gui_inverse.py`

The inverse design GUI solves the reverse problem: given measured composite
properties, find the constituent properties and/or microstructure that
produced them. It supports two models in sequence.

### The two-stage characterisation workflow

The recommended workflow separates concerns across two sequential inverse solves:

**Stage 1 — Elastic inverse**
Fix all fiber datasheet properties. Set matrix modulus and all microstructure
fields (orientation tensor, aspect ratio, mass fraction) to Free. Enter
measured E1, E2, G12, ν12 as targets. The solver recovers the matrix stiffness
and fiber orientation for that specific printer.

**Stage 2 — Thermoelastic inverse**
Click **Load from Card** to reload the Stage 1 result. The inferred
microstructure and matrix modulus auto-fill as Fixed. Set fiber CTE (f_cte1,
f_cte2) and matrix CTE (m_cte) to Free. Enter measured CTE11, CTE22 as targets.
Because CTE is a constituent (not microstructure) property, the result is stored
globally and reused across all printers that use the same fiber-polymer pair.

### Fixed vs Free inputs

Each input row has a toggle:

- **Fixed** — the value is known and held constant during optimisation.
  Enter the exact value.
- **Free** — an unknown to be estimated. Enter an initial guess. Optionally
  set lower (`lo`) and upper (`hi`) box bounds. The solver explores from the
  initial guess subject to those bounds.

At least one input must be set to Free. The solver enforces `a11 + a22 ≤ 1`
as a hard constraint when both orientation components are free.

### Target outputs and measurement noise

Check the outputs you want to target and enter the measured value. The `σ`
(sigma) field accepts the standard deviation of your measurement uncertainty:

| σ value | Loss behaviour |
|---|---|
| 0 | Standard squared error — penalises any deviation from the target |
| > 0 | ε-insensitive loss — deviations smaller than σ are not penalised; deviations beyond σ scale quadratically |

Using σ > 0 is recommended when your measurements have known noise or
repeatability uncertainty. It prevents the solver from overfitting to
measurement error and produces more physically plausible results.

### Solver options

Six optimisation methods are available:

| Method | Type | Best for |
|---|---|---|
| `lbfgs` | Gradient-based, no bounds | Smooth problems, fast convergence, good first choice |
| `lbfgsb` | Gradient-based, box bounds | When you have tight lo/hi bounds on free variables |
| `adam` | Gradient-based, stochastic | Rough or noisy loss landscapes where L-BFGS diverges |
| `differential_evolution` | Global, population-based | Multi-modal problems; requires bounds on all free variables |
| `dual_annealing` | Global, stochastic | When the landscape has many local minima |
| `basinhopping` | Global, random restarts | Robust global search; combines local L-BFGS hops |

**Additional solver settings:**

| Setting | Default | Meaning |
|---|---|---|
| Max iterations | 300 | Hard stop on optimisation iterations |
| Tolerance | 1×10⁻⁹ | Convergence criterion on gradient norm |
| Penalty weight | 10 000 | Multiplier for constraint violations (a11 + a22 ≤ 1) |
| Seed | 42 | Random seed for reproducibility with stochastic methods |
| Restarts | 1 | Number of random restarts; each uses a different initial point |

For most inverse problems, start with `lbfgs` at default settings. If the
solver converges to a poor result (high residual, physically implausible values),
try `lbfgsb` with tight bounds or switch to a global method with 5–10 restarts.

### Identifiability analysis

Before solving, click **Identifiability Check** to compute the Fisher
Information Matrix for your current input configuration. This answers the
question: *given the outputs I am targeting, how much information do my
measurements actually carry about each free variable?*

The tool displays:
- A ranked sensitivity bar chart: which free inputs are most identifiable
  from the chosen target outputs
- The FIM condition number: a high condition number (> 10⁴) indicates that
  some free variables are nearly unidentifiable from the current targets, and
  that small measurement noise will produce large uncertainty in the estimates
- Recommended actions if variables are poorly identifiable (add targets,
  tighten bounds, fix the problematic variable)

### Results and saving

After the solve completes, the results panel shows:
- Optimised values for all free variables
- Predicted outputs vs target outputs with percentage error per property
- Full predicted output vector

Click **Save to Card** to persist the result. The dialog lets you add to an
existing card or create a new one (specifying fiber, polymer, and printer).
The following are written to the database:
- Microstructure snapshot with per-field provenance (`inferred` for free
  fields solved by the optimiser, `inputted` for fixed fields)
- Inferred constituent property values (matrix modulus, CTEs)
- Composite property predictions
- A full inference run audit entry (all inputs, outputs, solver configuration,
  final loss)

Duplicate microstructure snapshots are never created: if you load a card and
re-run the thermoelastic inverse with the microstructure fixed (all fields
`inputted`), the existing snapshot is reused rather than duplicated.

---

## 7. Thermal Inverse Estimation

**Entry points:**
- `python gui_thermal_inverse.py` (standalone)
- **Thermal Inverse** button inside `gui_inverse.py`
- `python scripts/run_inverse_thermal.py` (CLI)

The thermal inverse recovers four constituent thermal conductivity parameters
from measured composite thermal conductivities (K11, K22, K33) at multiple
temperatures.

### Parametric constituent models

```
Polymer:  K_m(T) = p1 · √(T / T_ref) + p2     (T_ref = 1.0 °C)
Fiber:    K_f1   = l2                           (temperature-independent)
          K_f2   = l2 / t                       (temperature-independent)
```

| Parameter | Physical meaning | Typical range |
|---|---|---|
| `p1` | Temperature sensitivity of matrix conductivity | 0 – 7×10⁻³ W/(m·°C) |
| `p2` | Baseline matrix conductivity | 0 – 0.08 W/(m·°C) |
| `l2` | Fiber longitudinal conductivity | 1 – 20 W/(m·°C) |
| `t` | Fiber anisotropy ratio K_f1 / K_f2 | 1.01 – 6 |

From these four parameters the GUI also derives and displays:
- k_f1 = l2 (fiber longitudinal conductivity)
- k_f2 = l2 / t (fiber transverse conductivity)
- k_m at 25 °C = p1·√25 + p2 (room-temperature matrix conductivity)

### Input data format

Prepare a CSV or Excel file:

```
Temperature,K11,K22,K33
25,0.42,0.38,0.38
50,0.44,0.40,0.40
75,0.46,0.42,0.42
100,0.48,0.44,0.44
```

At least one of K11, K22, K33 must be present. Columns not present are excluded
from the loss automatically.

### Solver

The thermal inverse uses a multi-restart L-BFGS-B optimiser with physical box
bounds on all four parameters. Multiple restarts mitigate local minima. The
default is 10 restarts with seed 0.

### Fixed structural inputs

Before running, provide the microstructure:

| Field | Description |
|---|---|
| Aspect ratio | Fiber length / diameter |
| Mass fraction | Fiber mass fraction (0–1) |
| Fiber density | ρ_f in kg/m³ |
| Matrix density | ρ_m in kg/m³ |
| a11, a22, a12, a13, a23 | Fiber orientation tensor |

Click **Load from Card** to auto-fill these from a previously saved card
(Stage 1 elastic inverse result). This ensures the thermal inverse uses the
same microstructure that was inferred from mechanical measurements.

### Saving to card

Click **Save to Card** after a successful solve. The following are written:
- `k_f1`, `k_f2`, `k_m` (at 25 °C) as inferred constituent properties
  (globally reusable — stored with `print_config_id = NULL`)
- `p1`, `p2` as polymer parametric coefficients
- Full K vs T curve (K11, K22, K33 over all measured temperatures) to the
  `thermal_k_predictions` table — one row per save, cross-referenced by card

The existing microstructure snapshot is **reused** from the prior elastic
inverse run. The thermal inverse never re-infers microstructure, so no
duplicate snapshot is created.

---

## 8. Physics-Informed Transfer Learning

**Entry point:** `python gui_transfer.py`

Physics-informed transfer learning is the process of predicting composite
material properties on a **new printer** using constituent properties
characterised on a different printer — without running any new inverse
experiments on the new printer.

### The physical basis

Constituent properties (matrix modulus, fiber and matrix CTE, fiber and matrix
thermal conductivities) are intrinsic to the material chemistry. They do not
depend on how the part is printed. Only the **microstructure** — how the fibers
are oriented and how much fiber is present — changes between printers.

This separation means:

1. Characterise constituent properties on Printer A (Stages 1–3 above)
2. Determine the microstructure on Printer B — either by running a lightweight
   elastic inverse against a few mechanical measurements, or by entering the
   orientation tensor manually from EBSD or μCT data
3. Combine Printer B's microstructure with Printer A's constituent properties
   in the forward surrogate to predict all composite properties for Printer B

No new thermal or thermoelastic experiments are needed. The physics encoded
in the surrogate handles the combination.

### Transfer workflow in the GUI

#### Step 1 — Select source card

Choose the card from Printer A that contains the fully characterised
constituent properties (Stages 1–3 complete). The panel displays all resolved
constituent properties with their provenance:

| Property | Stage | Fallback if not inferred |
|---|---|---|
| Matrix modulus, Poisson's ratio | Stage 1 | Datasheet value |
| Fiber CTE (axial, transverse), Matrix CTE | Stage 2 | Datasheet value |
| k_f1, k_f2, k_m | Stage 3 | Computed from p1/p2 or datasheet |

#### Step 2 — Determine Printer B's microstructure

Two modes are available:

**Manual entry** — enter the orientation tensor (a11, a22, a12, a13, a23),
aspect ratio, and mass fraction directly. Use this when the microstructure is
known from independent measurement (μCT, EBSD) or from manufacturer
specification.

**Inverse estimation** — measure a small set of composite mechanical properties
on Printer B (E1, E2, or G12) and let the solver recover the microstructure.
The constituent properties from the source card are held Fixed; only the
microstructure fields are Free. This requires far fewer measurements than a
full characterisation because the constituent properties are already known.

The inverse estimation in transfer mode supports the same solver options as the
main inverse GUI (lbfgs, lbfgsb, adam, differential_evolution, dual_annealing,
basinhopping) with the same noise (σ) and bounds configuration.

#### Step 3 — Forward prediction

Select one or more models and click **Predict**:

**Elastic** — predicts E1, E2, E3, G12, G13, G23, ν12, ν13, ν23 for Printer B.

**Thermoelastic** — runs both elastic and thermoelastic surrogates. Mechanical
properties come from the elastic model; only the CTE outputs (CTE11–CTE23) come
from the thermoelastic model. This keeps the elastic properties consistent
across stages.

**Thermal** — sweeps 0–200 °C (50 temperature points by default). At each
temperature T, the matrix conductivity is computed as:

```
K_m(T) = p1 · √(T / T_ref) + p2
```

using p1 and p2 from the source card's inferred polymer constituent properties.
The thermal surrogate is evaluated at each T with the corresponding K_m(T),
producing full K11(T), K22(T), K33(T) curves. The results panel displays
values at 0, 25, 100, and 200 °C as a summary table.

#### Step 4 — Save to Card

Click **Save to Card** to attach the prediction results to a card for Printer B.
The dialog lets you add to an existing card or create a new one.

- For elastic / thermoelastic predictions: saved as `predicted` composite
  property values.
- For thermal predictions: the full K vs T curve is written to
  `thermal_k_predictions`, exactly as when saving from the thermal inverse GUI.
  Constituent properties (k_f1, k_f2, k_m, p1, p2) are also stored on the
  target card.

### What you need on the source card

For the transfer to work, the source card must have at minimum:
- A microstructure snapshot (from Stage 1 elastic inverse)
- Matrix modulus and Poisson's ratio (Stage 1)

For thermoelastic and thermal transfer predictions, it additionally needs:
- Fiber and matrix CTE (Stage 2)
- k_f1, k_f2, and either k_m directly or p1/p2 parametric coefficients (Stage 3)

If Stage 3 has not been run, the thermal conductivities fall back to the
polymer's datasheet `neat_k` value. If the datasheet value is also absent,
the thermal surrogate prediction is skipped.

### Example: transferring from Markforged X7 to Anisoprint

```
Source card:  CF-PESU / Markforged X7   (Stages 1–3 complete)
Target:       CF-PESU / Anisoprint

Step 1: Load source card → constituent props auto-resolve
         matrix_modulus = 4 100 MPa  [inferred]
         f_cte1 = 0.5 µ/K  [inferred]
         k_f1 = 9.8 W/m·K  [inferred]
         …

Step 2: Measure E1, E2 on the Anisoprint part
        Run inverse estimation → recovers a11=0.64, a22=0.18, mf=0.22
        (only 2 measurements needed; constituent properties are fixed)

Step 3: Predict Thermoelastic
        → E1=22.4 GPa, E2=6.1 GPa, CTE11=4.2 µ/K, CTE22=31.8 µ/K

        Predict Thermal (0–200 °C sweep)
        → K11(25°C)=0.58, K22(25°C)=0.44, K33(25°C)=0.44  [W/m·K]

Step 4: Save to Card → creates CF-PESU / Anisoprint card with full predictions
```

---

## 9. Database Design

### Three-layer separation

| Layer | Tables | Printer-dependent? |
|---|---|---|
| Constituent | `fibers`, `polymers`, `constituent_property_values` | No — intrinsic to material chemistry |
| Microstructure | `microstructure_snapshots` | Yes — determined by printer |
| Composite | `composite_property_values`, `experimental_measurements`, `thermal_k_predictions` | Yes — derived from both layers |

**Additional tables:** `print_configs` (cards), `printers`, `inference_runs`
(full audit trail), `property_preferences`.

### Unit conventions

All values in the database are stored in model units — no conversion is applied
on write or read. The unit manager (`core/unit_manager.py`) handles
display-unit conversion in the GUI layer only.

| Quantity | Model unit |
|---|---|
| Moduli (E, G) | MPa |
| Density | kg/m³ |
| CTE | 1/K |
| Conductivity | W/m·K |

### Thermal conductivity curves

K vs T prediction curves (K11, K22, K33 over a temperature range) are stored
in the dedicated `thermal_k_predictions` table — one row per save event,
cross-referenced by `print_config_id` and `inference_run_id`. This avoids
row explosion in `composite_property_values` and keeps the curve retrievable
as a single query.

---

## 10. Seeded Material Library

**Fibers**

| Name | Supplier | E1 (MPa) | k_f1 (W/m·K) |
|---|---|---|---|
| Carbon Fiber T300 | Toray | 230 000 | 10.5 |
| E-Glass | Owens Corning | 72 400 | 1.05 |
| AS4 | Hexcel | 231 000 | 12.83 |

**Polymers**

| Name | Supplier | E (MPa) | k (W/m·K) |
|---|---|---|---|
| PESU Ultrason | BASF | 3 600 | 0.26 |
| Epoxy 3501-6 | Hexcel | 4 200 | 0.17 |

Add new materials via **Manage Materials** in `gui.py` or `gui_inverse.py`,
or by editing `data/fibers.json` and `data/polymers.json` and re-running
`python db/init_db.py` (this resets the database).
