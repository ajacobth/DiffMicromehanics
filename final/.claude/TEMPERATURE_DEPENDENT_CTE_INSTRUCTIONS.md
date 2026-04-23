# Temperature-Dependent Thermoelastic Inverse — Implementation Plan

## Background

The existing thermoelastic surrogate maps scalar constituent CTEs to composite CTEs:

```
(f_cte1, f_cte2, m_cte, [15 structural inputs]) → (CTE11, CTE22, CTE33, ...)
```

No retraining is required. The surrogate can be evaluated pointwise at each temperature T
by substituting parametric CTE(T) model outputs as inputs. This mirrors exactly how
`core/inverse_thermal.py` handles temperature-dependent thermal conductivity.

---

## Step 1 — Define parametric constituent CTE(T) models

Create classes analogous to `PolymerConductivityModel` and `FiberConductivityModel`
in `inverse_thermal.py`.

**Polymer CTE** (isotropic, increases with temperature):
```
CTE_m(T) = a + b * T
```
Free parameters: `a` (offset), `b` (slope)

**Fiber CTE** (transversely isotropic; treat as weakly T-dependent or constant):
```
CTE_f1(T) = c0 + c1 * T    (axial)
CTE_f2(T) = d0 + d1 * T    (transverse)
```
Free parameters: `c0, c1, d0, d1` — or simplify to `c0, d0` if fiber is treated as
temperature-independent (fewer free parameters, more robust inversion).

Wrap these in a `ConstituentCTEParams` dataclass (mirror of `ConstituentParams` in
`inverse_thermal.py`) with `to_array()` / `from_array()` methods.

---

## Step 2 — Build a batched predictor

Mirror `make_batched_predictor` from `inverse_thermal.py`, wrapping the thermoelastic
surrogate (loaded via `load_forward("thermoelastic")`).

The thermoelastic surrogate has 19 inputs (vs thermal's 12) but the vmap pattern is
identical:

```python
import jax
batched = jax.jit(jax.vmap(fwd.predict_array))
```

The predictor accepts `X: np.ndarray, shape (N, 19)` and returns `Y: np.ndarray, shape (N, 15)`.

Output index map for thermoelastic (from `model_config.json` output_fields):
```
CTE11 → index 9, CTE22 → index 10, CTE33 → index 11
```

---

## Step 3 — Write `compute_composite_cte()`

Assemble the `(N, 19)` input matrix where:
- 16 structural inputs (`e1, e2, g12, f_nu12, f_nu23, ar, fiber_massfrac, fiber_density,
  matrix_modulus, matrix_poisson, matrix_density, a11, a22, a12, a13, a23`) are tiled
  constant across all N temperature rows
- `f_cte1`, `f_cte2`, `m_cte` vary row-by-row from the parametric models evaluated at T

Return columns `[CTE11(T), CTE22(T)]` (or CTE33 if measured).

Input field order must match `model_config.json` exactly:
```
e1, e2, g12, f_nu12, f_nu23, f_cte1, f_cte2, ar, fiber_massfrac,
fiber_density, matrix_modulus, matrix_poisson, matrix_density, m_cte,
a11, a22, a12, a13, a23
```

---

## Step 4 — Write the objective function

MSE between predicted CTE(T) curves and measured CTE vs T data:

```python
def objective_function(x, temperatures, CTE_data, predictor, fixed_inputs):
    params = ConstituentCTEParams.from_array(x)
    CTE_pred = compute_composite_cte(params, temperatures, predictor, fixed_inputs)
    loss = 0.0
    n_pts = 0
    for key, col in [("CTE11", 0), ("CTE22", 1)]:
        if CTE_data.get(key) is not None:
            loss  += float(np.sum((CTE_pred[:, col] - CTE_data[key]) ** 2))
            n_pts += len(temperatures)
    return loss / max(n_pts, 1)
```

---

## Step 5 — Write the multi-start L-BFGS-B solver

Mirror `run_inverse_estimation` from `inverse_thermal.py`:
- Random restarts within physical bounds for the CTE model parameters
- Scipy `minimize` with `method="L-BFGS-B"`
- Return `best_params, best_loss`

Physical bounds for CTE model parameters (SI units, 1/K):

| Parameter | Lo | Hi | Notes |
|---|---|---|---|
| a (m_cte offset) | 20e-6 | 120e-6 | Typical polymer range |
| b (m_cte slope) | 0.0 | 0.5e-6 | Per °C change |
| c0 (f_cte1 offset) | -2e-6 | 5e-6 | Axial fiber CTE |
| c1 (f_cte1 slope) | 0.0 | 0.05e-6 | Near-zero for stiff fibers |
| d0 (f_cte2 offset) | 5e-6 | 30e-6 | Transverse fiber CTE |
| d1 (f_cte2 slope) | 0.0 | 0.1e-6 | |

If fiber is fixed (temperature-independent), remove `c1` and `d1` to reduce to 4 free
parameters — same count as the thermal problem.

---

## Step 6 — Create the new file

Place all of the above in:
```
core/inverse_thermoelastic_td.py
```

This keeps it parallel to `core/inverse_thermal.py` and separate from the
single-temperature `core/inverse.py`.

Exports to expose:
```python
ConstituentCTEParams
make_batched_predictor_te    # wraps thermoelastic surrogate
compute_composite_cte
run_inverse_estimation_td
```

---

## Step 7 — Add a service wrapper (optional, for agent integration)

In `core/services/service_thermoelastic_td.py`:
- `run_thermoelastic_td_inverse(card_id, temperatures, CTE11_data, CTE22_data, **solver_kwargs)`
- Load fixed structural inputs from `service_cards.load_card_inputs(card_id)`
- Call `run_inverse_estimation_td`
- Return a result dict with recovered CTE model parameters and fit quality

This follows the same service-layer pattern as `service_forward.py` and `service_fim.py`.

---

## Step 8 — Add a GUI or agent tool (after core is validated)

Options:
- Extend `gui_thermal_inverse.py` with a "Thermoelastic TD" tab that accepts CTE vs T
  CSV input (same UX as thermal inverse)
- Add an agent tool `run_thermoelastic_td_inverse` in `agent/agent_tools.py` that calls
  the service wrapper (same pattern as existing inverse tools)

---

## Data requirements from the user

The user must supply:
- Measured `CTE11(T)` and/or `CTE22(T)` as arrays at N temperatures (CSV or manual entry)
- Temperatures in °C
- Fixed structural inputs: `e1, e2, g12, f_nu12, f_nu23, ar, fiber_massfrac, fiber_density,
  matrix_modulus, matrix_poisson, matrix_density, a11, a22, a12, a13, a23`
  (these can come from a loaded material card)

---

## Key assumption

The surrogate was trained with constant-CTE data. Feeding it CTE(T) evaluated
pointwise at each temperature is valid — each call is a standard homogenization
evaluation with a different set of constituent properties. The micromechanics
relationship does not itself change with temperature.
