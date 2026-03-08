# Instructions: Add ε-Insensitive Loss

## Background

The current loss in `_loss_fn` is:
```python
errs = jnp.stack([(y[out_idx[k]] - t) ** 2 for k, t in prob.target_outputs.items()])
loss = jnp.sum(errs)
```
This was recently updated to divide by `sig_out` (output normalisation std):
```python
errs = jnp.stack([((y[out_idx[k]] - t) / sig_out[out_idx[k]]) ** 2
                   for k, t in prob.target_outputs.items()])
loss = jnp.sum(errs)
```
The goal is to add a dead zone around each residual so that L-BFGS does not
chase noise in poorly-identified parameter directions.

---

## What to implement

### The new loss term

For each target output k, replace:
```python
((y[out_idx[k]] - t) / sig_out[out_idx[k]]) ** 2
```
With:
```python
(jnp.maximum(0.0, jnp.abs(y[out_idx[k]] - t) - sigma_k) / sig_out[out_idx[k]]) ** 2
```

Where `sigma_k` is the user-supplied experimental measurement noise for output k,
in **original physical units** (GPa for moduli, dimensionless for Poisson ratios).
When `sigma_k = 0` this reduces exactly to the current loss — fully backward compatible.

---

## Changes to `problem.json`

Add two new optional top-level keys. Do not remove or rename anything existing.

```json
"sigmas": {
  "E1":  0.30,
  "E2":  0.20,
  "E3":  0.20
},
"solver": {
  "use_epsilon_loss": false,
  "epsilon_scale":    1.0
}
```

- `sigmas`: measurement noise per target output in physical units. Any key not
  present defaults to `0.0` (standard loss for that term).
- `use_epsilon_loss` (bool, default `false`): master on/off switch.
- `epsilon_scale` (float, default `1.0`): multiplies all sigma values uniformly.
  Setting to `0.0` recovers standard loss regardless of sigma values.

---

## Changes to `inverse.py`

### 1. Read sigmas in `run()`

After the line that loads `targets`, add:

```python
sigmas_dict      = prob_dict.get("sigmas", {})
use_eps_loss     = bool(solver_cfg.get("use_epsilon_loss", False))
epsilon_scale    = float(solver_cfg.get("epsilon_scale", 1.0))

# ordered to match prob.target_outputs key order
sigmas_list = [
    float(sigmas_dict.get(k, 0.0)) * epsilon_scale
    for k in targets.keys()
]
```

If `use_epsilon_loss` is True, print a summary before solving:
```
ε-insensitive loss enabled (epsilon_scale=1.0):
  E1  : ε = 0.3000  (normalised: 0.0412)
  E2  : ε = 0.2000  (normalised: 0.0389)
  E3  : ε = 0.2000  (normalised: 0.0485)
```
The normalised value is `sigma_k / sig_out[out_idx[k]]` — show it so the
user can sanity-check the tube width relative to the loss scale.

### 2. Add arguments to `_loss_fn`

New signature (add with safe defaults so all existing call sites still work):
```python
def _loss_fn(vec64, predict_array, prob, in_idx, out_idx, n_inputs,
             sig_out,                        # already exists
             penalty=1e4,
             sigmas=None,                    # NEW — list of floats, physical units
             use_epsilon_loss=False):        # NEW — bool
```

Replace the `errs` computation with:
```python
errs = []
for i, (k, t) in enumerate(prob.target_outputs.items()):
    pred  = y[out_idx[k]]
    scale = sig_out[out_idx[k]]
    if use_epsilon_loss and sigmas is not None and sigmas[i] > 0.0:
        raw    = jnp.abs(pred - t)
        excess = jnp.maximum(0.0, raw - sigmas[i])
        errs.append((excess / scale) ** 2)
    else:
        errs.append(((pred - t) / scale) ** 2)
loss = jnp.sum(jnp.stack(errs))
```

### 3. Thread through `_solve` and `_solve_global`

Pass `sigmas_list` and `use_eps_loss` into the `loss` lambda the same way
`sig_out` is already threaded. The lambda inside `_solve` becomes:

```python
loss = lambda v: _loss_fn(
    v, predict_array, prob, in_idx, out_idx, n_inputs,
    sig_out, penalty, sigmas_list, use_eps_loss
)
```

---

## Changes to `gui_inverse.py`

### 1. σ input per target output

In the section where the user enters experimental target values, add a second
input field beside each target labelled **"σ (noise)"**. This is where the user
types the experimental standard deviation in physical units (the scatter across
their repeat tensile tests).

When the user saves or runs, write these values into `problem.json` under
`"sigmas"`.

### 2. Solver options panel

Add two controls to the solver options section:
- Checkbox: **"Use ε-insensitive loss"** → writes `use_epsilon_loss` to `"solver"`
- Numeric input: **"ε scale"** (default 1.0, min 0.0) → writes `epsilon_scale`
  to `"solver"`

---

## Files to modify

| File | Change |
|------|--------|
| `inverse.py` | New args on `_loss_fn`, read sigmas in `run()`, thread through `_solve` |
| `gui_inverse.py` | σ column in targets table, checkbox + scale input in solver panel |
| `problem.json` | Add `"sigmas"` dict and `use_epsilon_loss` / `epsilon_scale` to solver |

Do not modify `forward.py`, `gui.py`, or `field_labels.json`.

---

## Rules

- `sigma_k = 0` must reduce to exactly the current loss for that term.
- All new `problem.json` fields are optional — existing configs work unchanged.
- ε is always specified and stored in **physical units**. The `/sig_out`
  normalisation happens inside `_loss_fn`, never outside it.
