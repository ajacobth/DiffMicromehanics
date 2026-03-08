"""
Run this inside your DiffMicromechanics environment to check whether
the NN surrogate has spurious nu_m sensitivity in E1, E2, E3.

Usage:
    python check_nu_sensitivity.py
"""
import numpy as np
import jax
import jax.numpy as jnp

# ── Load your model the same way gui_identifiability.py does ──────────────
# Replace this with however you load ForwardModel in your codebase
from forward import load_forward
import json

with open("problem.json") as f:
    prob = json.load(f)

model = load_forward("models/")   # adjust path as needed

# ── Build nominal x ────────────────────────────────────────────────────────
fixed = prob["fixed_inputs"]
x_np  = np.zeros(len(model.input_fields), dtype=np.float32)
for k, v in fixed.items():
    if k in model.in_idx:
        x_np[model.in_idx[k]] = float(v)

x0 = jnp.array(x_np)

# ── Compute dE1/dnu_m, dE2/dnu_m, dE3/dnu_m via JAX autodiff ─────────────
nu_m_idx = model.in_idx["matrix_poisson"]
E1_idx   = model.out_idx["E1"]
E2_idx   = model.out_idx["E2"]
E3_idx   = model.out_idx["E3"]

def f_nu(nu_m_val):
    x = x0.at[nu_m_idx].set(nu_m_val)
    y = model.predict_array(x)
    return jnp.array([y[E1_idx], y[E2_idx], y[E3_idx]])

J_nu = jax.jacobian(f_nu)(x0[nu_m_idx])
y0   = model.predict_array(x0)

print("Sensitivity of axial moduli to matrix_poisson (nu_m):")
print(f"  dE1/dnu_m = {float(J_nu[0]):10.2f}  MPa/unit")
print(f"  dE2/dnu_m = {float(J_nu[1]):10.2f}  MPa/unit")
print(f"  dE3/dnu_m = {float(J_nu[2]):10.2f}  MPa/unit")
print()
print("Predicted values at nominal:")
print(f"  E1={float(y0[E1_idx]):.0f}, E2={float(y0[E2_idx]):.0f}, E3={float(y0[E3_idx]):.0f} MPa")
print()

threshold = 50.0  # MPa per unit change in nu_m
if max(abs(float(J_nu[i])) for i in range(3)) > threshold:
    print(f"⚠  Surrogate has significant nu_m -> E coupling (>{threshold} MPa/unit).")
    print("   This is a training artefact — the physical model has zero coupling.")
    print("   The FIM will report nu_m as MARGINAL instead of POOR.")
    print("   This makes nu_m look more identifiable than it truly is.")
    print()
    print("   Options:")
    print("   1. Re-train surrogate with physics-informed constraint dE/dnu_m = 0")
    print("   2. Trust the physics: manually set dE/dnu_m = 0 in fim.py for E1/E2/E3")
    print("   3. Accept MARGINAL as a conservative (safe) over-estimate of identifiability")
else:
    print("✓  Surrogate correctly has near-zero nu_m sensitivity for axial moduli.")
    print("   POOR classification is correct — something else is causing MARGINAL.")
