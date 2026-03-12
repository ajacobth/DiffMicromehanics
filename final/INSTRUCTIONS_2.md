# Instructions: Identifiability GUI & FIM Improvements

## Files to modify
| File | Changes |
|------|---------|
| `gui_identifiability.py` | σ visibility, fixed param editor, recommendation display |
| `fim.py` | Consistent σ for recommendations, show λ_min not ratio |

Do **not** modify `forward.py`, `inverse.py`, `gui.py`, `gui_inverse.py`, or `problem.json`.

---

## Change 1 — σ fields only visible when output is checked

### Problem
Currently σ entry fields are shown for all outputs regardless of whether the
checkbox is ticked. This clutters the UI with irrelevant fields.

### Fix in `gui_identifiability.py` — `_build_selection_panels`

For each output row, store a reference to `sig_frame` and the checkbox variable.
Bind a trace on the `BooleanVar` so that when unchecked, `sig_frame` is hidden
via `grid_remove()`, and when checked it is shown via `grid()`.

```python
# After creating cb and sig_frame for each output:
sig_frame_ref = sig_frame   # keep reference

def _toggle_sigma(name=name, sf=sig_frame_ref, var=var):
    if var.get():
        sf.grid()
    else:
        sf.grid_remove()

var.trace_add("write", lambda *_: _toggle_sigma())
_toggle_sigma()  # apply initial state (unchecked → hidden)
```

Store `sig_frame_ref` in a dict `self._sigma_frames[name] = sig_frame_ref` so
it can be shown/hidden from other methods (e.g. `_on_help`).

In `_on_help`, after setting `var.set(name in target_in_problem)`, call
`_toggle_sigma()` (or equivalent) so the σ field appears for pre-checked outputs.

---

## Change 2 — Editable fixed parameter values panel

### Problem
The FIM is always evaluated with fixed parameters taken from `problem.json`
`fixed_inputs`. The user cannot check identifiability in a different region of
the parameter space (e.g. higher fiber fraction, different orientation regime)
without editing the JSON file manually.

### Fix in `gui_identifiability.py`

#### 2a. Add a collapsible "Fixed Parameter Values" panel

Add this panel **below** the free/target selection panels (between
`_build_selection_panels` and `_build_controls`), i.e. as a new row in
`_main_frame`.

```
┌─ Fixed Parameter Values ──────────────────────────────┐
│  ► Expand to edit                                       │
│                                                         │
│  (when expanded:)                                       │
│  Fiber Modulus Longitudinal (MPa)   [ 240000.0 ]       │
│  Fiber Modulus Transverse (MPa)     [  15000.0 ]       │
│  ...all non-free model inputs...                        │
└─────────────────────────────────────────────────────────┘
```

- Use a `ttk.LabelFrame` with text `"Fixed Parameter Values"`.
- Inside, place a toggle button `"► Expand to edit"` / `"▼ Collapse"`.
- When expanded, show one row per model input field that is NOT currently
  selected as free. Each row has a label (display name from `field_labels.json`)
  and a `tk.Entry` pre-filled with the value from `problem.json fixed_inputs`.
- Store the entry `tk.StringVar` in `self._fixed_override_vars: dict[str, tk.StringVar]`.
- When a free variable checkbox is toggled, refresh the fixed panel so the
  newly freed variable disappears from it and vice versa.

#### 2b. Use edited values when building `x_template` in `_on_run`

Replace the current block that reads from `self._problem["fixed_inputs"]`:

```python
# CURRENT (reads only from problem.json):
fixed = {k: float(v) for k, v in self._problem.get("fixed_inputs", {}).items() ...}
x_np = np.zeros(...)
for k, v in fixed.items():
    if k in self.model.in_idx:
        x_np[self.model.in_idx[k]] = v
# Set free vars to midpoint of their bounds
for k in free_inputs:
    ...
    x_np[self.model.in_idx[k]] = (lo + hi) / 2.0
```

With:

```python
# NEW: start from problem.json, then overlay user edits
fixed_base = {k: float(v) for k, v in self._problem.get("fixed_inputs", {}).items()
              if not str(k).startswith("_")}
x_np = np.zeros(len(self.model.input_fields), dtype=np.float32)

for k, v in fixed_base.items():
    if k in self.model.in_idx:
        x_np[self.model.in_idx[k]] = v

# Apply user overrides from the Fixed Parameter Values panel
for k, svar in self._fixed_override_vars.items():
    if k not in free_inputs and k in self.model.in_idx:
        raw = svar.get().strip()
        try:
            x_np[self.model.in_idx[k]] = float(raw)
        except ValueError:
            pass  # keep problem.json value if invalid

# Free vars: midpoint of bounds (overwritten at each LHS sample in fim.py)
for k in free_inputs:
    if k in bounds_raw and k in self.model.in_idx:
        lo, hi = bounds_raw[k]
        x_np[self.model.in_idx[k]] = (lo + hi) / 2.0

x_template = jnp.array(x_np, dtype=jnp.float32)
```

---

## Change 3 — Consistent σ for recommendations (fix in `fim.py`)

### Problem
The recommendation ranking compares current targets (which may have σ=0,
triggering the 2% fallback) against candidate experiments (which always use
the 2% fallback because the user has not entered σ for them). This is
inconsistent: a current target with σ=0 gets treated as noiseless (infinite
information), making any candidate look relatively worse.

The ranking should use a **uniform noise assumption** for all outputs when
computing candidate FIM contributions, so the comparison is fair.

### Fix in `fim.py` — `_precompute_all_output_contributions`

The function already receives `sigmas` (the user-entered values). For outputs
with `sigma <= 0`, it uses 2% of predicted value. This is correct for
candidates. But the same function is also used to compute the baseline FIM
(current targets) inside `compute_normalised_fim`, where σ=0 results in
noiseless treatment.

**Rule:** whenever σ is not entered (≤ 0) for any output — whether current
target or candidate — always use the **2% fallback**. Never treat σ=0 as
noiseless. The 2% fallback is already implemented; the fix is to ensure it
is applied consistently in `compute_normalised_fim` as well.

In `compute_normalised_fim`, change:

```python
# CURRENT — σ=0 falls through, causing noiseless treatment:
sigma_i = float(sigmas.get(k, 0.0))
if sigma_i <= 0:
    y_pred = float(predict_array(...)[out_idx[k]])
    sigma_i = max(abs(y_pred) * 0.02, 1e-12)
```

Confirm this block already exists (it does — lines 140–150 in the current
`fim.py`). It is correct. **No change needed in `compute_normalised_fim`.**

The actual inconsistency is in the GUI: the user can type `0.0` in a σ field
and this is passed as `sigma=0.0` to `fim.py`, which then triggers the 2%
fallback. This is fine. **Document this behaviour** with a tooltip or note
near the σ field:

> "Leave blank or enter 0 to use 2% of predicted value as noise estimate."

Update the σ field placeholder text (the `Entry` widget) to show `"auto"` as
placeholder when empty, so the user understands a default is being used.

---

## Change 4 — Recommendation display: show λ_min not improvement ratio

### Problem
The current display shows `"improves worst direction by {imp:.1f}×"`. This is:
- Numerically unstable when the baseline λ_min ≈ 0 (ratios like 2,847,392× appear)
- Uninformative: the ratio doesn't tell the user whether the result is WELL/MARGINAL/POOR
- Misleading: a large ratio may still leave a parameter POOR

### Fix in `gui_identifiability.py` — `_show_results`

Replace the recommendation text block:

```python
# CURRENT:
if has_issues and recs:
    self._rec_text.insert("end", "To improve identifiability, consider adding:\n\n")
    for rank, rec in enumerate(recs, 1):
        name   = rec["name"]
        imp    = rec["improvement_factor"]
        disp   = _label(self._labels, "outputs", name)
        line   = f"  {rank}. {name:8s}  ({disp})\n"
        line  += f"        improves worst direction by {imp:.1f}×"
        if imp < 1.01:
            line += "  (no improvement)"
        line += "\n\n"
        self._rec_text.insert("end", line)
```

With:

```python
# NEW: show λ_min and status tag instead of improvement ratio
def _lam_status(lam):
    if lam >= 100:  return "WELL"
    elif lam >= 1:  return "MARGINAL"
    else:           return "POOR"

if has_issues and recs:
    self._rec_text.insert("end",
        "Best additional experiments (ranked by information gain):\n"
        "λ_min is the worst-direction eigenvalue after adding the experiment.\n"
        "λ ≥ 100 = WELL  |  1 ≤ λ < 100 = MARGINAL  |  λ < 1 = POOR\n\n")
    for rank, rec in enumerate(recs, 1):
        name     = rec["name"]
        lam_new  = rec["new_min_eigenvalue"]
        disp     = _label(self._labels, "outputs", name)
        status   = _lam_status(lam_new)
        tag      = {"WELL": "✓", "MARGINAL": "~", "POOR": "✗"}[status]
        line     = f"  {rank}. {name:<8s}  {disp}\n"
        line    += f"        adding this → λ_min = {lam_new:.1f}  [{tag} {status}]\n\n"
        self._rec_text.insert("end", line)
```

Also update **Plot 3** in `_show_advanced_plots` to use `new_min_eigenvalue`
on the x-axis instead of `improvement_factor`. Add horizontal reference lines
at x=1 (POOR boundary) and x=100 (WELL boundary). Change x-axis label to
`"λ_min after adding experiment (log scale)"`.

```python
# In Plot 3:
lams_r = [r["new_min_eigenvalue"] for r in recs]  # was: imps_r

ax3.barh(y_pos, lams_r, color=colors3)             # was: imps_r
ax3.axvline(1.0,   color="red",    linestyle="--", label="Identifiable threshold (λ=1)")
ax3.axvline(100.0, color="orange", linestyle="--", label="Well-identified threshold (λ=100)")
ax3.set_xlabel("λ_min after adding experiment (log scale)")
ax3.set_title("Recommended Next Experiment — λ_min in worst-identified direction")
```

---

## Change 5 — Always show recommendations (not only when issues exist)

### Problem
Recommendations are currently hidden when all parameters are WELL. But the user
may still want to know which experiment is most informative, or which is
redundant (adds no information).

### Fix in `gui_identifiability.py` — `_show_results`

Change the condition from `if has_issues and recs:` to always show
recommendations, but with different header text:

```python
if recs:
    if has_issues:
        header = "Best additional experiments to fix identifiability issues:\n"
    else:
        header = "All parameters are well-identified. Additional experiments ranked by information gain:\n"
    header += ("λ_min is the worst-direction eigenvalue after adding the experiment.\n"
               "λ ≥ 100 = WELL  |  1 ≤ λ < 100 = MARGINAL  |  λ < 1 = POOR\n\n")
    self._rec_text.insert("end", header)
    for rank, rec in enumerate(recs, 1):
        ...  # same as Change 4 above
```

---

## Summary of what each change fixes

| # | What was wrong | What it fixes |
|---|---------------|---------------|
| 1 | σ fields shown for unchecked outputs | Clean UI — σ only appears when relevant |
| 2 | Fixed params locked to problem.json | User can probe identifiability at any point in parameter space |
| 3 | σ=0 treated as noiseless for current targets | Fair comparison between current and candidate experiments |
| 4 | Improvement ratio unstable, uninformative | λ_min + status tag is stable and directly interpretable |
| 5 | Recommendations hidden when all WELL | Always shows ranked experiments for curiosity / redundancy analysis |

---

## Testing checklist

After implementing, verify the following manually:

1. **σ visibility**: check E1, confirm σ field appears. Uncheck E1, confirm σ field disappears. Recheck — σ field reappears with its previous value intact.

2. **Fixed params panel**: expand the panel. Change `fiber_massfrac` from 0.20 to 0.30. Run check. Result should differ from default because the Jacobian is evaluated at the new fiber fraction.

3. **σ=0 consistency**: check E1 and E3, leave σ blank (auto). Check that the result is MARGINAL for a22 (not WELL as would happen if σ=0 was treated as noiseless). The 2% fallback should be applied.

4. **Recommendation display**: with only E1 checked and a11+a22 free, confirm recommendations show E2 ranked #1 with a large λ_min and WELL tag, not G12.

5. **Plot 3**: confirm x-axis is λ_min, reference lines appear at 1 and 100, bars for POOR candidates are visibly below the red line.