# Inverse Estimation of Constituent Thermal Conductivities
## Instructions for Claude Code

---

## Overview

This task implements gradient-based inverse estimation of **fiber and matrix (polymer) constituent thermal conductivities** as functions of temperature, given measured composite thermal conductivities (K11, K22, K33) at multiple temperatures.

The approach follows Thomas et al. (2024), *Computer Methods in Applied Mechanics and Engineering*, but replaces Bayesian MCMC with deterministic gradient-based optimization (scipy `minimize` with L-BFGS-B or similar).

---

## Files to Produce

You must produce exactly **two Python files**:

1. `inverse_estimation_temp.py` — contains all model definitions, parameterizations, and the core optimization logic  
2. `run_inverse_temp.py` — the user-facing entry point that loads data and calls the estimator

---

## Input Data Format

The user provides a **CSV or Excel file** with the following columns (case-insensitive):

| Column Name | Description |
|-------------|-------------|
| `Temperature` | Temperature in °C |
| `K11` | Composite thermal conductivity in the 1-direction (W/m·°C) |
| `K22` | Composite thermal conductivity in the 2-direction (W/m·°C) |
| `K33` | Composite thermal conductivity in the 3-direction (W/m·°C) |

- The file can be `.csv` or `.xlsx`
- Column names should be matched **case-insensitively**
- Missing K columns are allowed — the optimization will use only the available ones
- A sample data file `sample_data.csv` should also be created as a usage example

---

## Constituent Property Parameterizations

These come directly from the paper (Section 4.2). **Implement exactly these functional forms.**

### Polymer thermal conductivity (isotropic scalar, temperature-dependent):

```
K_p(T) = p1 * sqrt(T / T_ref) + p2
```

where:
- `T` is temperature in °C
- `T_ref = 1.0` (reference temperature, fixed)
- `p1`, `p2` are scalar parameters to be estimated (both must be ≥ 0)

### Fiber thermal conductivity — longitudinal direction (temperature-dependent):

```
K_f_long(T) = l1 * T + l2
```

where:
- `l1`, `l2` are scalar parameters to be estimated
- `l1` can be small (near zero) and positive
- `l2` is the dominant term (room-temperature fiber conductivity, typically O(10) W/m·°C for carbon fiber)

### Fiber thermal conductivity — transverse direction (temperature-dependent):

```
K_f_trans(T) = K_f_long(T) / t
```

where:
- `t` is a scalar parameter to be estimated (must be > 1, since transverse conductivity < longitudinal for carbon fiber)

### Summary of parameters to estimate:
- `p1` — polymer conductivity scaling (≥ 0)
- `p2` — polymer conductivity offset (≥ 0)  
- `l1` — fiber longitudinal conductivity temperature slope (≥ 0)
- `l2` — fiber longitudinal conductivity intercept (> 0)
- `t`  — fiber anisotropy ratio (> 1)

Total: **5 scalar parameters**

---

## Micromechanics Model

Use the **Mori-Tanaka + Voigt homogenization** already implemented in the existing codebase (the `DiffMicroMechanics` or equivalent module in the repository). This model takes:

- `K_p` — polymer conductivity (scalar at a given temperature)
- `K_f_long` — fiber longitudinal conductivity (scalar)
- `K_f_trans` — fiber transverse conductivity (scalar)
- `A` — second-order fiber orientation tensor, diagonal: `[a11, a22, a33]`
- `fiber_aspect_ratio` — average fiber aspect ratio (scalar)
- `fiber_volume_fraction` — fiber volume fraction (scalar)

and returns `[K11_pred, K22_pred, K33_pred]` — predicted composite conductivities.

**If the micromechanics model cannot be imported**, implement a simplified Mori-Tanaka scalar version inline as a fallback (see note below).

---

## Objective Function

Minimize the **mean squared error** across all temperatures and all available directions:

```
loss = sum over temperatures t:
         sum over directions i in {1, 2, 3} where K_ii data is available:
           (K_ii_pred(t) - K_ii_data(t))^2
```

Normalize by the number of (temperature, direction) data points.

---

## Optimization

- Use `scipy.optimize.minimize` with method `'L-BFGS-B'`
- Apply **bounds** consistent with the physical constraints:
  - `p1`: [0, 1]
  - `p2`: [0, 1]
  - `l1`: [0, 0.1]
  - `l2`: [1, 100]
  - `t`:  [1.01, 50]
- Use **multiple random restarts** (at least 10) to avoid local minima. Pick the result with the lowest final loss.
- Log each restart's final loss to the console.

---

## Outputs

After optimization, the code should:

1. **Print** the estimated parameter values to the console
2. **Print** the final loss (MSE)
3. **Save** a CSV file `estimated_conductivities.csv` with columns:
   - `Temperature`
   - `K_polymer` — estimated polymer conductivity at each temperature
   - `K_fiber_long` — estimated fiber longitudinal conductivity
   - `K_fiber_trans` — estimated fiber transverse conductivity
   - `K11_pred`, `K22_pred`, `K33_pred` — predicted composite conductivities
   - `K11_data`, `K22_data`, `K33_data` — measured values (NaN if not provided)
4. **Save** a plot `fit_comparison.png` showing predicted vs. measured K11, K22, K33 vs. temperature, and a second subplot showing the inferred constituent conductivities vs. temperature.
5. **Save** The finctions fit for fiber and polymer conductivities as a function of temperature
---

## `inverse_estimation_temp.py` — Required Contents

This file must contain:

1. **`PolymerConductivityModel`** — class or function for `K_p(T; p1, p2)`
2. **`FiberConductivityModel`** — class or function for `K_f_long(T; l1, l2)` and `K_f_trans(T; l1, l2, t)`
3. **`ConstituentParams`** — a dataclass holding `[p1, p2, l1, l2, t]` with a method to pack/unpack to/from a numpy array
4. **`compute_composite_conductivity(params, temperatures, orientation_tensor, aspect_ratio, vf)`** — calls the micromechanics model at each temperature and returns predicted `[K11, K22, K33]` arrays
5. **`objective_function(x, temperatures, K_data, orientation_tensor, aspect_ratio, vf)`** — computes MSE loss; `x` is the flat parameter vector; `K_data` is a dict with keys `'K11'`, `'K22'`, `'K33'` (each may be None or a numpy array)
6. **`run_inverse_estimation(temperatures, K_data, orientation_tensor, aspect_ratio, vf, n_restarts=10)`** — runs the multi-start L-BFGS-B optimization and returns the best `ConstituentParams` and final loss

No user interaction or file I/O should be in this file.

---

## `run_inverse_temp.py` — Required Contents

This file is the entry point. It must:

1. Accept the data file path as a **command-line argument** (using `argparse`) in additon to the config file being loaded, the rest of the detials
should be in the config file:
   ```
   python run_inverse_temp.py --data my_data.csv 
   ```
2. Also accept:
   - `--a11`, `--a22`, `--a33` — diagonal components of the orientation tensor (must sum to 1; default: `0.77, 0.16, 0.07`)
   - `--aspect_ratio` — fiber aspect ratio (default: `20`)
   - `--vf` — fiber volume fraction (default: `0.174`, corresponding to ~25 wt% carbon fiber in PESU)
   - `--n_restarts` — number of optimization restarts (default: `10`)
   - `--output_dir` — directory for output files (default: current directory)
3. Load the data file (auto-detect CSV vs Excel by extension)
4. Normalize column names to lowercase
5. Extract temperature and available K columns
6. Call `run_inverse_estimation(...)` from `inverse_estimation.py`
7. Save outputs as described above
8. Print a summary table to stdout

---

## Dependencies

Use only standard scientific Python packages:
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

----

## Sample Data File

Create `sample_data.csv` with synthetic data representative of CF-PESU at 25 wt% carbon fiber (from the paper's Fig. 5):

```
Temperature,K11,K22,K33
25,1.8,0.65,0.30
50,1.85,0.67,0.31
75,1.88,0.69,0.32
100,1.90,0.70,0.33
125,1.92,0.71,0.34
150,1.93,0.72,0.35
175,1.95,0.73,0.36
195,1.96,0.74,0.37
```

---

## Code Style Requirements

- All functions must have **docstrings** explaining inputs, outputs, and units
- Use `numpy` vectorized operations wherever possible (avoid Python loops over temperatures)
- Add inline comments explaining each parameterization choice, referencing the paper equation numbers
- Use `if __name__ == '__main__':` guard in `run_inverse.py`

---


