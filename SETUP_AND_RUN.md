# Setup and Run Guide
## Composite Micromechanics Surrogate – Forward & Inverse GUIs

This guide is written for someone who is new to Python and just wants to run
the two graphical tools:

- **Forward GUI** (`gui.py`) – predict composite material properties from
  microstructure inputs.
- **Inverse GUI** (`gui_inverse.py`) – find the microstructure that achieves
  a set of target material properties.

Follow the section for your operating system.

---

## Table of Contents

1. [What You Need Before Starting](#1-what-you-need-before-starting)
2. [Windows – CPU (no GPU required)](#2-windows--cpu-no-gpu-required)
3. [Windows – GPU (NVIDIA CUDA, optional)](#3-windows--gpu-nvidia-cuda-optional)
4. [Mac – CPU (any Mac)](#4-mac--cpu-any-mac)
5. [Mac – GPU (Apple Silicon only)](#5-mac--gpu-apple-silicon-only)
6. [Linux – CPU](#6-linux--cpu)
7. [Linux – GPU (NVIDIA CUDA, optional)](#7-linux--gpu-nvidia-cuda-optional)
8. [Test Your Environment](#8-test-your-environment)
9. [Run the Forward GUI](#9-run-the-forward-gui)
10. [Run the Inverse GUI](#10-run-the-inverse-gui)
11. [Common Errors and Fixes](#11-common-errors-and-fixes)

---

## 1. What You Need Before Starting

### Get the code

You need the **entire repository**, not just the `final/` folder. The GUIs
load model code from a sibling folder called `EL_surrogate/`. If you only have
`final/`, they will not work.

**Download the full repository** from GitHub (ask your supervisor for the link)
by clicking the green **Code** button → **Download ZIP**, then unzip it.
Or if you have Git installed:

```
git clone <repository-url>
```

After downloading, you should have a folder structure like this:

```
DiffMicromehanics/
├── EL_surrogate/           ← required (loaded automatically)
├── THEL_surrogate/
├── final/                  ← this is where you will work
│   ├── gui.py              ← forward GUI
│   ├── gui_inverse.py      ← inverse GUI
│   ├── test_setup.py       ← run this to check your setup
│   └── models/
│       ├── elastic/
│       └── thermoelastic/
└── ...
```

> **Important:** Do NOT move `final/` out of the main `DiffMicromehanics/`
> folder. The code needs the parent folder structure to find its model files.

---

## 2. Windows – CPU (no GPU required)

> These instructions use CPU only. The models run fast enough on a modern CPU.
> GPU is entirely optional.

### Step 1 – Install Miniconda

Miniconda is a lightweight Python installer that also manages separate
"environments" so packages for this project do not conflict with other software.

1. Go to: **https://docs.conda.io/en/latest/miniconda.html**
2. Download the **Windows 64-bit** installer (the `.exe` file).
3. Run the installer. When asked:
   - Accept the license agreement.
   - Choose "Just Me" (recommended).
   - Leave the install location at the default (`C:\Users\YourName\miniconda3`).
   - **Check** "Add Miniconda3 to my PATH" **OR** leave it unchecked and use
     the **Anaconda Prompt** app that gets installed. Either way works.
4. Click Install, then Finish.

After installation, open **Anaconda Prompt** (search for it in the Start menu).
All commands below should be typed in this window.

---

### Step 2 – Create a virtual environment

A **virtual environment** is like a clean, isolated box for Python and its
packages. Installing packages inside the environment does not affect the rest
of your computer, and activating/deactivating it is like opening or closing
that box.

```bash
conda create -n diffmech python=3.10 -y
```

- `conda create` makes a new environment.
- `-n diffmech` gives it the name `diffmech` (you can choose any name).
- `python=3.10` pins the Python version (3.10 is stable and tested).
- `-y` automatically says "yes" to all prompts.

This takes about 1–2 minutes.

---

### Step 3 – Activate the environment

```bash
conda activate diffmech
```

Your prompt will change from `(base)` to `(diffmech)`. This tells you the
environment is active. **Every time you open a new Anaconda Prompt, you must
run this command again before running the GUIs.**

To deactivate (go back to the default environment):

```bash
conda deactivate
```

> Think of `activate` as "opening the box" and `deactivate` as "closing it".
> Packages you install while the box is open stay inside that box and do not
> affect other projects.

---

### Step 4 – Navigate to the project folder

Use `cd` ("change directory") to move into the `final/` folder.
Replace the path below with the actual location on your computer:

```bash
cd C:\Users\YourName\Downloads\DiffMicromehanics\final
```

> **Tip:** You can drag the `final` folder from Windows Explorer into the
> Anaconda Prompt window to paste its path automatically.

To confirm you are in the right place:

```bash
dir
```

You should see files like `gui.py`, `gui_inverse.py`, `test_setup.py`, and the
`models` folder.

---

### Step 5 – Install JAX (CPU version)

JAX is the machine-learning library the models use. On Windows CPU:

```bash
pip install "jax[cpu]==0.4.26"
```

This downloads and installs JAX and its companion library (`jaxlib`) together.
It may take 2–5 minutes.

---

### Step 6 – Install the remaining packages

```bash
pip install `
  numpy==1.26.2 `
  scipy==1.11.4 `
  matplotlib==3.8.2 `
  flax==0.8.3 `
  optax==0.1.7 `
  jaxopt==0.8.5 `
  chex==0.1.85 `
  ml-collections==0.1.1 `
  ml-dtypes==0.3.2 `
  orbax-checkpoint==0.4.8 `
  absl-py==2.0.0 `
  opt-einsum==3.3.0 `
  msgpack==1.0.7 `
  dm-tree==0.1.8 `
  etils==1.5.2 `
  toolz==0.12.0
```

> **Note:** The backtick `` ` `` at the end of each line is the Windows
> PowerShell line-continuation character. If you are using **Anaconda Prompt**
> (not PowerShell), use `^` instead, or paste all packages on a single line.

**Single-line version (works in all Windows terminals):**

```bash
pip install numpy==1.26.2 scipy==1.11.4 matplotlib==3.8.2 flax==0.8.3 optax==0.1.7 jaxopt==0.8.5 chex==0.1.85 ml-collections==0.1.1 ml-dtypes==0.3.2 orbax-checkpoint==0.4.8 absl-py==2.0.0 opt-einsum==3.3.0 msgpack==1.0.7 dm-tree==0.1.8 etils==1.5.2 toolz==0.12.0
```

---

### Step 7 – Verify with the test script

Jump to [Section 8 – Test Your Environment](#8-test-your-environment).

---

## 3. Windows – GPU (NVIDIA CUDA, optional)

> Only follow this if you have an NVIDIA graphics card and want faster
> computation. Skip to [Section 4](#4-mac--cpu-any-mac) if you are on Mac.

Repeat Steps 1–4 from Section 2 first (install Miniconda, create env,
activate, navigate to folder).

### Install CUDA-enabled JAX

First check which CUDA version your GPU supports:

```bash
nvidia-smi
```

Look for the "CUDA Version" in the top-right corner of the output.

**For CUDA 12.x:**

```bash
pip install "jax[cuda12_pip]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**For CUDA 11.x:**

```bash
pip install "jax[cuda11_pip]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then install the remaining packages exactly as in Step 6 of Section 2.

> **Note:** If JAX detects your GPU, you will see it listed when you run
> `python test_setup.py`. The inverse GUI forces CPU regardless (it sets
> `JAX_PLATFORM_NAME=cpu` internally), so GPU mainly speeds up the forward
> GUI.

---

## 4. Mac – CPU (any Mac)

These instructions work on both Intel Macs and Apple Silicon (M1/M2/M3) Macs
when you want CPU-only execution.

### Step 1 – Install Miniconda

1. Go to: **https://docs.conda.io/en/latest/miniconda.html**
2. Download the correct installer:
   - **Apple Silicon (M1/M2/M3):** download the `arm64` `.pkg` or `.sh` file.
   - **Intel Mac:** download the `x86_64` `.pkg` or `.sh` file.
3. Run the `.pkg` installer and follow the prompts.
   If you downloaded the `.sh` file, open **Terminal** and run:
   ```bash
   bash ~/Downloads/Miniconda3-latest-MacOSX-arm64.sh
   ```
   (Replace `arm64` with `x86_64` if on Intel.)
4. When asked "Do you wish to initialize Miniconda3?", type `yes`.
5. Close and reopen Terminal.

---

### Step 2 – Create the environment

Open **Terminal** and run:

```bash
conda create -n diffmech python=3.10 -y
```

---

### Step 3 – Activate the environment

```bash
conda activate diffmech
```

Your prompt will change to show `(diffmech)`. Remember: run this every time
you open a new Terminal window before using the GUIs.

To deactivate:

```bash
conda deactivate
```

---

### Step 4 – Navigate to the project folder

```bash
cd ~/Downloads/DiffMicromehanics/final
```

Adjust the path to match where you actually put the folder. To confirm:

```bash
ls
```

You should see `gui.py`, `gui_inverse.py`, `test_setup.py`, and `models/`.

---

### Step 5 – Install JAX (CPU version)

```bash
pip install "jax[cpu]==0.4.26"
```

---

### Step 6 – Install the remaining packages

```bash
pip install \
  numpy==1.26.2 \
  scipy==1.11.4 \
  matplotlib==3.8.2 \
  flax==0.8.3 \
  optax==0.1.7 \
  jaxopt==0.8.5 \
  chex==0.1.85 \
  ml-collections==0.1.1 \
  ml-dtypes==0.3.2 \
  orbax-checkpoint==0.4.8 \
  absl-py==2.0.0 \
  opt-einsum==3.3.0 \
  msgpack==1.0.7 \
  dm-tree==0.1.8 \
  etils==1.5.2 \
  toolz==0.12.0
```

---

### Step 7 – Verify

Jump to [Section 8 – Test Your Environment](#8-test-your-environment).

---

## 5. Mac – GPU (Apple Silicon only)

> Only for M1/M2/M3/M4 Macs. Intel Macs do not support this.

Complete Steps 1–6 from Section 4 (CPU install) first, then add the Metal GPU
backend on top:

```bash
pip install jax-metal==0.1.0
```

After this, JAX will use the GPU automatically. You can verify with
`python test_setup.py` – it will report the backend as `metal`.

> **Note:** `jax-metal` is experimental. If you encounter crashes or unexpected
> results, uninstall it and fall back to CPU:
> ```bash
> pip uninstall jax-metal
> ```

---

## 6. Linux – CPU

### Step 1 – Install Miniconda

Open a terminal and run:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts. When asked to initialize conda, type `yes`.
Close and reopen your terminal.

---

### Step 2 – Create and activate the environment

```bash
conda create -n diffmech python=3.10 -y
conda activate diffmech
```

---

### Step 3 – Install tkinter (Linux only)

On Linux, tkinter (the GUI toolkit) is not always bundled with Python.
Install it via your system package manager **before activating conda**
(open a new terminal tab):

```bash
# Ubuntu / Debian:
sudo apt-get install python3-tk

# Fedora / RHEL:
sudo dnf install python3-tkinter

# Arch Linux:
sudo pacman -S tk
```

---

### Step 4 – Navigate to the project folder

```bash
cd ~/Downloads/DiffMicromehanics/final
```

---

### Step 5 – Install JAX and remaining packages

```bash
pip install "jax[cpu]==0.4.26"

pip install \
  numpy==1.26.2 \
  scipy==1.11.4 \
  matplotlib==3.8.2 \
  flax==0.8.3 \
  optax==0.1.7 \
  jaxopt==0.8.5 \
  chex==0.1.85 \
  ml-collections==0.1.1 \
  ml-dtypes==0.3.2 \
  orbax-checkpoint==0.4.8 \
  absl-py==2.0.0 \
  opt-einsum==3.3.0 \
  msgpack==1.0.7 \
  dm-tree==0.1.8 \
  etils==1.5.2 \
  toolz==0.12.0
```

---

## 7. Linux – GPU (NVIDIA CUDA, optional)

Complete Steps 1–4 from Section 6 first, then install the CUDA-enabled JAX.

Check your CUDA version:

```bash
nvidia-smi
```

**CUDA 12.x:**

```bash
pip install "jax[cuda12_pip]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**CUDA 11.x:**

```bash
pip install "jax[cuda11_pip]==0.4.26" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then install the remaining packages as in Section 6 Step 5.

---

## 8. Test Your Environment

Once you have finished installing packages, run the test script to check
that everything is working.

Make sure you are:
1. In the `final/` folder (run `cd .../DiffMicromehanics/final`)
2. The `diffmech` environment is activated (your prompt shows `(diffmech)`)

Then run:

```bash
python test_setup.py
```

### What to expect – all PASS

```
=== 1. Python version ===
  [PASS]  Python 3.10.x
  [PASS]  Python >= 3.9

=== 2. Core packages ===
  [PASS]  numpy 1.26.2
  [PASS]  scipy 1.11.4
  [PASS]  matplotlib 3.8.2

=== 3. JAX ===
  [PASS]  jax 0.4.26
  [INFO]  JAX backend: cpu
  [INFO]  Running on CPU (normal if you have no GPU)
  [PASS]  jax basic computation (sum)
  [PASS]  jax JIT compilation
  [PASS]  jax autodiff (grad)

=== 4. ML packages ===
  [PASS]  flax 0.8.3
  [PASS]  optax 0.1.7
  [PASS]  jaxopt 0.8.5
  [PASS]  chex 0.1.85
  [PASS]  ml-collections
  [PASS]  orbax-checkpoint

=== 5. GUI (tkinter) ===
  [PASS]  tkinter (GUI toolkit)

=== 6. Folder structure ===
  [PASS]  final/ folder found
  [PASS]  EL_surrogate/ folder found (sibling of final/)
  [PASS]  models/elastic/ exists
  [PASS]  models/thermoelastic/ exists
  ...

=== 7. Load surrogate models (end-to-end test) ===
  [PASS]  Load + predict 'elastic' (16 inputs → 9 outputs)
  [PASS]  Load + predict 'thermoelastic' (19 inputs → 15 outputs)

============================================================
  All checks PASSED!  You are ready to run the GUIs.

  Forward GUI : python gui.py
  Inverse GUI : python gui_inverse.py
============================================================
```

### What to do if something FAILS

Each `[FAIL]` line includes a hint showing exactly what to install or fix.
After fixing, run `python test_setup.py` again.

---

## 9. Run the Forward GUI

The Forward GUI lets you specify composite microstructure inputs and instantly
predicts the resulting material properties.

### Launch it

Make sure your environment is activated, then from the `final/` folder:

```bash
python gui.py
```

A window titled **"Composite Surrogate Predictor"** will open.

---

### Step-by-step usage

#### 1. Select the model

At the top of the window you will see two radio buttons:

- **Elastic** – predicts 9 elastic properties (Young's moduli, shear moduli,
  Poisson's ratios).
- **Thermoelastic** – predicts all 9 elastic properties plus 6 thermal
  expansion coefficients (15 outputs total).

Select one and click **Load Model**.

The first time you load a model it may take 5–15 seconds while JAX compiles the
neural network (this is called JIT compilation). You will see a spinning status
message. Subsequent runs in the same session are instant.

A green status line confirms the model loaded:
```
Elastic loaded  (16 in / 9 out)
```

#### 2. Fill in the inputs

The left panel fills with input fields. Enter a numerical value in every field.

| Input symbol | What it represents | Typical units |
|---|---|---|
| `e1` | Fiber axial Young's modulus | MPa |
| `e2` | Fiber transverse Young's modulus | MPa |
| `g12` | Fiber shear modulus | MPa |
| `f_nu12` | Fiber major Poisson's ratio | dimensionless |
| `f_nu23` | Fiber transverse Poisson's ratio | dimensionless |
| `ar` | Fiber aspect ratio | dimensionless |
| `fiber_massfrac` | Fiber mass fraction | dimensionless (0–1) |
| `fiber_density` | Fiber density | kg/m³ |
| `matrix_modulus` | Matrix Young's modulus | MPa |
| `matrix_poisson` | Matrix Poisson's ratio | dimensionless |
| `matrix_density` | Matrix density | kg/m³ |
| `a11` | Fiber orientation tensor component A₁₁ | dimensionless |
| `a22` | Fiber orientation tensor component A₂₂ | dimensionless |
| `a12` | Fiber orientation tensor component A₁₂ | dimensionless |
| `a13` | Fiber orientation tensor component A₁₃ | dimensionless |
| `a23` | Fiber orientation tensor component A₂₃ | dimensionless |

For the **thermoelastic** model, three extra fiber CTE fields appear:
`f_cte1` (axial, 1/K), `f_cte2` (transverse, 1/K), and `m_cte` (matrix CTE,
1/K).

> **Orientation tensor constraint:** The components must satisfy
> `a11 + a22 ≤ 1.0` and all values must be between 0 and 1.

**Example values for a carbon-fibre / epoxy composite:**

| Field | Value |
|---|---|
| e1 | 240000 |
| e2 | 15000 |
| g12 | 28000 |
| f_nu12 | 0.20 |
| f_nu23 | 0.40 |
| ar | 20.0 |
| fiber_massfrac | 0.20 |
| fiber_density | 1780 |
| matrix_modulus | 3100 |
| matrix_poisson | 0.37 |
| matrix_density | 1280 |
| a11 | 0.6 |
| a22 | 0.1 |
| a12 | 0.0 |
| a13 | 0.0 |
| a23 | 0.0 |

#### 3. Predict

Click **Predict** (or press **Enter** in any input field).

The right panel updates with the predicted composite properties in display
units:

| Output | Unit displayed |
|---|---|
| E1, E2, E3 | GPa |
| G12, G13, G23 | GPa |
| nu12, nu13, nu23 | dimensionless |
| CTE11 … CTE23 | µ/K (micro-strain per Kelvin) |

#### 4. Identifiability Check (optional)

Click **Identifiability Check** in the top bar to open a sensitivity analysis
tool that shows which inputs most influence each output.

---

## 10. Run the Inverse GUI

The Inverse GUI solves the reverse problem: you specify the material properties
you want, and the solver finds the microstructure inputs that achieve them.

### Launch it

```bash
python gui_inverse.py
```

A window titled **"Composite Surrogate – Inverse Design"** will open.

---

### Step-by-step usage

#### 1. Load a model

Select **Elastic** or **Thermoelastic** at the top and click **Load Model**.
Wait for the green confirmation message.

#### 2. Configure inputs (left panel)

Each input row has two modes:

- **Fixed** – you know this value and it will not change during optimisation.
  Enter the value in the box.
- **Free** – this is an unknown that the solver will adjust to hit the targets.
  Enter an **initial guess** in the box. Optionally set `lo` and `hi` bounds.

**Rules:**
- You must enter a value (or initial guess) for every input.
- At least one input must be set to **Free**.
- For `a11` + `a22`: if both are Free, the solver automatically enforces
  `a11 + a22 ≤ 1.0`.

#### 3. Configure targets (right panel)

Check the checkbox next to each output you want to target. Enter the target
value in the entry box.

> **Units for target entry:** E/G values are in **MPa** (not GPa), Poisson's
> ratios are dimensionless, and CTE values are in **1/K** (not µ/K).
> The display column labelled `(MPa)`, `(1/K)` etc. shows the correct unit.

Optionally enter a noise value `σ` (standard deviation of measurement noise).
When σ > 0, the solver uses an ε-insensitive loss that tolerates small
deviations.

#### 4. Choose a solver method

In the solver bar at the bottom:

| Method | When to use |
|---|---|
| `lbfgs` | Default. Fast gradient-based, best for smooth problems. |
| `lbfgsb` | Like lbfgs but respects box bounds. Use when you set lo/hi. |
| `adam` | Slower but more robust when the loss landscape is rough. |
| `differential_evolution` | Global search. Requires bounds on all free vars. |
| `dual_annealing` | Global search. Requires bounds on all free vars. |
| `basinhopping` | Global search with random restarts. |

For most cases, start with `lbfgs` (default). If it does not converge, try
`lbfgsb` with bounds, or a global method.

**Other solver settings:**

| Setting | Default | Meaning |
|---|---|---|
| Max iter | 300 | Maximum optimisation iterations |
| Tol | 1e-9 | Convergence tolerance |
| Penalty | 10000 | Penalty weight for violated constraints |
| Seed | 42 | Random seed (for reproducibility with global methods) |

#### 5. Optional: Add a tag

In the top bar, you can type a tag such as `machine:CAMRI batch:A name:test1`.
Tags are saved in the result file to help you identify which run produced which
result. The `name:...` tag is required for saving results.

#### 6. Click SOLVE

The solver runs in the background. The status bar shows "Solving…" while it
runs. When finished, the **Results** panel at the bottom shows:

- Optimised values of the free variables.
- Comparison of predicted vs. target outputs (with percentage errors).
- All predicted outputs.

#### 7. Export results

If a `name:...` tag was set, click **Export Results** to save the result as a
JSON file. Use **Browse…** to choose the save folder.

---

### Example inverse problem

Goal: find the matrix modulus and orientation tensor that give
E1 ≈ 15 420 MPa and E2 ≈ 5 140 MPa.

| Input | Mode | Value / Guess |
|---|---|---|
| e1 | Fixed | 240000 |
| e2 | Fixed | 15000 |
| g12 | Fixed | 28000 |
| f_nu12 | Fixed | 0.20 |
| f_nu23 | Fixed | 0.40 |
| ar | Fixed | 20.0 |
| fiber_massfrac | Fixed | 0.20 |
| fiber_density | Fixed | 1780 |
| matrix_modulus | **Free** | 3100 (initial guess) |
| matrix_poisson | Fixed | 0.37 |
| matrix_density | Fixed | 1280 |
| a11 | **Free** | 0.6 (initial guess) |
| a22 | **Free** | 0.1 (initial guess) |
| a12 | Fixed | 0.0 |
| a13 | Fixed | 0.0 |
| a23 | Fixed | 0.0 |

Targets (in MPa):

| Output | Target |
|---|---|
| E1 | 15420 |
| E2 | 5140 |

Solver: `lbfgs`, Max iter: 300, Tol: 1e-9.

---

## 11. Common Errors and Fixes

### "ModuleNotFoundError: No module named 'jax'"

The environment is not activated, or jax was not installed.

```bash
conda activate diffmech
pip install "jax[cpu]==0.4.26"
```

### "FileNotFoundError: Model directory not found: .../models/elastic"

You ran the script from the wrong folder, OR you moved `final/` out of the
main repository folder.

```bash
# Make sure you are in the right place:
cd /path/to/DiffMicromehanics/final
python gui.py
```

### "No module named 'NN_surrogate'" or "No module named 'models'"

The `EL_surrogate/` folder is missing. You likely only downloaded `final/`.
Download the full repository.

### "ModuleNotFoundError: No module named '_tkinter'"

On Linux, tkinter is not installed:

```bash
sudo apt-get install python3-tk   # Ubuntu / Debian
```

Then re-create your conda environment (tkinter must be available when conda
creates the env, or use the system Python).

Alternatively, install tkinter through conda:

```bash
conda install -c conda-forge tk
```

### The GUI window is very small / text is cut off

The GUIs are designed for a monitor of at least 1280 × 800 pixels. You can
resize the window by dragging its edges, and all panels scroll.

### "WARNING: All log messages before absl::InitializeLog() is called..."

This is a harmless message from the `absl` library. You can safely ignore it.

### The first prediction is slow (5–30 seconds)

JAX compiles ("JIT-compiles") the neural network the first time it runs.
Subsequent predictions in the same session are instant. This is normal.

### "CUDA error" or JAX reports GPU but crashes

Fall back to CPU by adding this line before running:

**Windows/Linux:**
```bash
set JAX_PLATFORM_NAME=cpu   # Windows
export JAX_PLATFORM_NAME=cpu  # Linux/Mac
python gui.py
```

Or uninstall the GPU backend:

```bash
pip uninstall jax-metal        # Mac
# For CUDA, reinstall CPU version:
pip install "jax[cpu]==0.4.26"
```

---

*If you encounter an error not listed here, copy the full error message and
send it to the project supervisor.*
