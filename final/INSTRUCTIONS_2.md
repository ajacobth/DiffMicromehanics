# Instructions: Identifiability Analysis GUI

## Goal

Create a new standalone GUI tab or window called **"Identifiability Check"**.
The user selects which outputs they plan to measure (target variables) and
which inputs they want to infer (free variables), then runs an identifiability
analysis powered by `fim.py`. The GUI tells them clearly whether their chosen
experiment set is sufficient to identify their chosen free variables, which
free variables are problematic, and which additional experiments to run — in
priority order. An optional **Advanced** section shows FIM eigenvalue plots
for users who want deeper insight.

This is a pre-inference planning tool. It does not run the optimisation.

---

## FIM approach: use the normalised FIM

Do NOT use the raw FIM. Use the **normalised FIM** where:
- Each Jacobian row is divided by the measurement noise `sigma_i` (physical units)
- Each Jacobian column is multiplied by the parameter range `(hi_j - lo_j)`

```
J_norm[i,j] = J[i,j] * (hi_j - lo_j) / sigma_i
```

This makes the FIM eigenvalues **dimensionless** and gives a clean threshold:
- `lambda < 1`  → parameter direction is **POOR** — cannot be identified within
  the parameter range given measurement noise
- `1 <= lambda < 100`  → **MARGINAL**
- `lambda >= 100` → **WELL** identified

This is better than the raw FIM because Em (GPa) and nu_m (dimensionless) have
very different scales, which distorts raw eigenvalue comparisons. The normalised
FIM treats all parameters and outputs fairly.

---

## Update `fim.py`

Replace or augment the existing `compute_fim` with a normalised version.
Add the following function:

### `compute_normalised_fim(predict_array, x_template, free_inputs, free_indices, target_outputs, out_idx, sigmas, bounds, N_samples=200)`

```python
def compute_normalised_fim(predict_array, x_template, free_inputs,
                            free_indices, target_outputs, out_idx,
                            sigmas, bounds, N_samples=200):
    """
    Normalised FIM where eigenvalues are dimensionless.

    J_norm[i,j] = J[i,j] * param_range[j] / sigma_i

    eigenvalue < 1   -> POOR   (cannot identify within param range)
    eigenvalue < 100 -> MARGINAL
    eigenvalue >= 100 -> WELL

    Parameters
    ----------
    sigmas : dict {output_name: float} measurement noise in physical units.
             If a key is missing, uses 1.0 as fallback.
    bounds : dict {param_name: (lo, hi)}
    """
    target_out_indices = [out_idx[k] for k in target_outputs.keys()]
    n_free      = len(free_inputs)
    F           = np.zeros((n_free, n_free))
    samples     = sample_parameter_space(free_inputs, bounds, N=N_samples)
    free_idx_arr = jnp.array(free_indices)

    param_ranges = np.array([bounds[k][1] - bounds[k][0]
                              for k in free_inputs])

    for theta in samples:
        x = x_template.at[free_idx_arr].set(
            jnp.array(theta, dtype=x_template.dtype))
        J = compute_jacobian(predict_array, x,
                             free_indices, target_out_indices)  # (n_exp, n_free)

        for i, k in enumerate(target_outputs.keys()):
            sigma_i = float(sigmas.get(k, 1.0))
            if sigma_i <= 0:
                sigma_i = 1.0
            # normalise: scale by param range / measurement noise
            j_norm = J[i, :] * param_ranges / sigma_i
            F     += np.outer(j_norm, j_norm)

    return F / max(len(samples), 1)
```

Also add a helper for the relative sensitivity matrix (used in the heatmap):

### `compute_relative_sensitivity(predict_array, x_nominal, free_indices, all_out_indices, param_names, output_names)`

```python
def compute_relative_sensitivity(predict_array, x_nominal, free_indices,
                                  all_out_indices, param_names, output_names):
    """
    S[i,j] = |dyi/dthj| * |thj / yi|

    Returns S as np.ndarray shape (n_outputs, n_free).
    Used for the heatmap showing which outputs are sensitive to which params.
    """
    J     = compute_jacobian(predict_array, x_nominal,
                             free_indices, all_out_indices)
    theta = np.array(x_nominal)[free_indices]
    y     = np.array(predict_array(x_nominal))[all_out_indices]
    S     = np.abs(J) * np.abs(theta[np.newaxis, :]) / (np.abs(y[:, np.newaxis]) + 1e-12)
    return S
```

---

## New GUI file: `gui_identifiability.py`

Create this as a standalone window/tab that can be launched from the main
`gui.py` menu (add a button or menu item: **"Identifiability Check"**).

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Identifiability Check                                       │
├───────────────────────┬─────────────────────────────────────┤
│  Free Variables       │  Target Outputs (experiments)        │
│  (what to infer)      │  (what you will measure)            │
│                       │                                     │
│  [ ] Em               │  [ ] E1    σ: [0.30] GPa            │
│  [ ] nu_m             │  [ ] E2    σ: [0.20] GPa            │
│  [x] a11              │  [x] E3    σ: [0.20] GPa            │
│  [x] a22              │  [ ] G12   σ: [0.15] GPa            │
│                       │  [ ] G13   σ: [0.15] GPa            │
│                       │  [ ] G23   σ: [0.15] GPa            │
│                       │  [ ] nu21  σ: [0.015]               │
│                       │  [ ] nu13  σ: [0.015]               │
│                       │  [ ] nu23  σ: [0.015]               │
│                       │                                     │
│       N samples: [200]│  [ ] Advanced plots                 │
│                       │                                     │
│       [ Run Check ]   │                                     │
├───────────────────────┴─────────────────────────────────────┤
│  Results                                                    │
│  ─────────────────────────────────────────────────────────  │
│  (populated after Run Check)                                │
└─────────────────────────────────────────────────────────────┘
```

### Left panel — Free Variables

Checkboxes for every input field that appears in `model.in_idx` and is not
a fixed fiber property. Read available free variable names from
`field_labels.json` for display labels. When the user checks a variable,
read its bounds from `problem.json` to use in the FIM computation.

### Right panel — Target Outputs

Checkboxes for every output field in `model.output_fields`. Next to each
checkbox, show a small numeric input for σ (measurement noise, physical units).
Pre-populate σ values from `problem.json["sigmas"]` if they exist, otherwise
leave blank. The user must fill in at least a rough σ — if left blank, default
to 1.0 with a tooltip: "Enter measurement standard deviation from repeat tests".

Show output names using `field_labels.json` labels.

### Controls

- **N samples** numeric input (default 200): how many parameter space samples
  to use for the Bayesian FIM average.
- **Advanced plots** checkbox: when ticked, show the advanced section after
  running (described below).
- **Run Check** button: triggers the analysis.

---

## Results section (always shown)

After clicking Run Check, populate the results section with:

### 1. Per-parameter status table

```
Parameter   Status      CR Uncertainty    Note
─────────────────────────────────────────────────────────────
Em          ✗ POOR      —                Cannot be identified
nu_m        ✗ POOR      —                Cannot be identified
a11         ✓ WELL      ±0.012           Reliably identified
a22         ✓ WELL      ±0.009           Reliably identified
```

- Status uses color: green = WELL, orange = MARGINAL, red = POOR
- CR Uncertainty = `sqrt(F_inv[j,j])` in physical parameter units
  (units come from `field_labels.json`). Show "—" if POOR (CR bound is
  meaningless when F is singular).
- Note column: plain English — "Reliably identified", "Marginally identified",
  "Cannot be identified".

### 2. Recommended experiments (only shown if any POOR or MARGINAL)

Show a ranked list of experiments NOT currently selected, ordered by how much
each one improves the worst eigenvalue (E-optimality criterion):

```
To improve identifiability, consider adding:

  1. nu13   — improves worst direction by 45×     [Panel A, no new specimens]
  2. G13    — improves worst direction by 12×
  3. nu21   — improves worst direction by 8×
```

The improvement factor is `lambda_min(F_new) / lambda_min(F_current)`.
Include a brief note where the experiment comes from if it can be inferred
from field labels (e.g. "Panel A" outputs vs "Panel B" outputs).

Show at most 5 recommendations.

### 3. Overall verdict (one line, prominent)

```
✓  Your experiment set can reliably identify all selected free variables.
```
or
```
⚠  2 free variable(s) cannot be reliably identified with this experiment set.
   See recommendations below.
```

Place this at the TOP of the results section in large text, green or red.

---

## Advanced section (shown only when "Advanced plots" is ticked)

This section appears below the results when the checkbox is ticked.
It contains three plots rendered with matplotlib embedded in the GUI.

### Plot 1: FIM Eigenvalue Spectrum

Bar chart (log scale y-axis) of the normalised FIM eigenvalues, sorted
ascending. Each bar is labeled with the dominant parameter direction
(the parameter name that eigenvector is most aligned with).

- Draw a horizontal dashed red line at `lambda = 1` labeled "Identifiability threshold"
- Draw a horizontal dashed orange line at `lambda = 100` labeled "Well-identified threshold"
- Bars below the red line are red, between lines are orange, above are green
- Title: "FIM Eigenvalue Spectrum — bars below red line indicate poor identifiability"

### Plot 2: Relative Sensitivity Heatmap

Heatmap showing `|dyi/dthj| * |thj/yi|` for all active target outputs
(rows) vs all free parameters (cols). Use a yellow-orange-red colormap.
Annotate each cell with the numeric value.

- Rows = active target outputs (experiments selected by user)
- Cols = free parameters selected by user
- Title: "Relative Sensitivity — darker = stronger coupling between output and parameter"
- This plot explains intuitively WHY certain experiments help: if a row has a
  dark cell in the Em column, that experiment is sensitive to Em.

### Plot 3: Eigenvalue Improvement per Candidate Experiment

Horizontal bar chart showing, for each experiment NOT currently selected,
how much it would improve the minimum FIM eigenvalue if added. Sort by
improvement descending. Highlight the top recommendation in a different color.

- X-axis: improvement factor (log scale)
- Y-axis: experiment names
- Title: "Recommended Next Experiment — improvement to worst-identified direction"
- Draw a vertical dashed line at improvement = 1 (no improvement)

---

## Integration into main `gui.py`

Add a button to the main GUI toolbar or menu:

```python
btn_identifiability = tk.Button(toolbar, text="Identifiability Check",
                                 command=open_identifiability_window)
```

The window opens independently and reads the current `problem.json` and
loaded model to pre-populate its fields.

---

## Integration into `fim.py`

Add these two functions used by the GUI:

### `run_identifiability_check(predict_array, x_template, free_inputs, free_indices, target_outputs, out_idx, sigmas, bounds, all_output_names, sig_out, N_samples=200)`

Single entry point called by the GUI. Returns a structured dict:

```python
{
    "status": {           # per free parameter
        "Em":   "POOR",
        "nu_m": "POOR",
        "a11":  "WELL",
        "a22":  "WELL",
    },
    "cramer_rao_std": {   # None if POOR
        "Em":   None,
        "nu_m": None,
        "a11":  0.012,
        "a22":  0.009,
    },
    "eigenvalues":    np.ndarray,   # normalised, ascending
    "eigenvectors":   np.ndarray,
    "dominant_params": list of str,
    "condition_number": float,
    "recommendations": [            # ranked by E-optimality improvement
        {"name": "nu13", "improvement_factor": 45.2},
        {"name": "G13",  "improvement_factor": 12.1},
        ...
    ],
    "sensitivity_matrix": np.ndarray,   # (n_targets, n_free), relative sensitivity
    "sensitivity_output_names": list,
    "sensitivity_param_names":  list,
    "F": np.ndarray,                # normalised FIM, for advanced plots
}
```

### `rank_candidate_experiments(predict_array, x_template, free_inputs, free_indices, current_targets, out_idx, sigmas, bounds, all_output_names, N_samples=200)`

Returns list of dicts sorted by E-optimality improvement:
```python
[
    {"name": "nu13", "improvement_factor": 45.2, "new_min_eigenvalue": 65.6},
    {"name": "G13",  "improvement_factor": 12.1, "new_min_eigenvalue": 17.3},
    ...
]
```

---

## Files to create / modify

| File | Action |
|------|--------|
| `fim.py` | **MODIFY** — add `compute_normalised_fim`, `compute_relative_sensitivity`, `run_identifiability_check`, `rank_candidate_experiments` |
| `gui_identifiability.py` | **CREATE** — full identifiability GUI window |
| `gui.py` | **MODIFY** — add "Identifiability Check" button that opens the new window |

Do not modify `inverse.py`, `forward.py`, or `field_labels.json`.

---

## Rules

- Use the **normalised FIM** throughout — not the raw FIM.
- The threshold for POOR is `lambda < 1`, MARGINAL is `1 <= lambda < 100`,
  WELL is `lambda >= 100`. These are dimensionless and physically meaningful
  for the normalised FIM.
- σ values left blank by the user default to `1.0` internally — never crash.
- Advanced plots only render when the checkbox is ticked — do not compute
  them otherwise (they are slow for large N_samples).
- The entire analysis must complete in under 10 seconds for N_samples=200.
  If it takes longer, reduce N_samples or cache the Jacobian computation.
- Use `jax.jacobian` for the Jacobian — never finite differences.
- Read parameter bounds from `problem.json`. If bounds are missing for a
  free variable, show an error: "Bounds required for identifiability analysis."
