"""inverse_model.py â€==“ bounded Adam / L-BFGS / L-BFGS-B

* Handles any jaxopt version (lower/upper, lower_bounds/upper_bounds, or
  bounds=(lo,hi)).
* Casts optimisation variables to **float64** while keeping the NN in
  float32 â†’ avoids XLA triangular_solve bug seen on some builds.
"""

from __future__ import annotations
import os
from typing import Mapping, Sequence, NamedTuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxopt import LBFGS, LBFGSB
import ml_collections

from NN_surrogate.utils import restore_checkpoint
import models

# -----------------------------------------------------------------------------
# Feature maps (must match training)
# -----------------------------------------------------------------------------

INPUT_FIELD_NAMES = [
    "k_f1", "k_f2", "k_m", "ar_f",
    "w_f", "rho_f", "rho_m",
    "a11", "a22", "a12", "a13", "a23", # a33 = 1 - a11 - a22
]

# The size of the inputs is automatically foudn during the code execution

#k_f1	k_f2	k_m	ar_f	w_f	rho_f	rho_m	a11	a22	a12	a13	a23

# k11	k12	k13	k22	k23	k33
OUTPUT_FIELD_NAMES = [
    "k11", "k12", "k13", "k22", "k23",
    "k33"]

IN_IDX  = {n: i for i, n in enumerate(INPUT_FIELD_NAMES)}
OUT_IDX = {n: i for i, n in enumerate(OUTPUT_FIELD_NAMES)}

# -----------------------------------------------------------------------------
InputKey = Union[int, str]
OutputKey = Union[int, str]
class InverseProblem(NamedTuple):
    fixed_inputs: Mapping[InputKey, float]
    free_inputs:  Sequence[InputKey]
    target_outputs: Mapping[OutputKey, float]

# -----------------------------------------------------------------------------
# Predictor
# -----------------------------------------------------------------------------

def create_predictor(cfg: ml_collections.ConfigDict, workdir: str):
    model = models.MICRO_SURROGATE_L2(cfg)
    params = restore_checkpoint(model.state, os.path.join(workdir, "ckpt", cfg.wandb.name))["params"]

    stats = np.load(os.path.join(workdir, "normalization_stats.npz"))
    mu_in, sig_in   = map(jnp.array, (stats["input_mean"],  stats["input_std"]))
    mu_out, sig_out = map(jnp.array, (stats["target_mean"], stats["target_std"]))

    @jax.jit
    def fwd(x):
        xb = x[None] if x.ndim == 1 else x
        y_s = model.u_net(params, (xb - mu_in) / sig_in)
        y   = y_s * sig_out + mu_out
        return y[0] if x.ndim == 1 else y
    return fwd

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_in  = lambda k: k if isinstance(k,int) else IN_IDX[k]
_out = lambda k: k if isinstance(k,int) else OUT_IDX[k]

def assemble_x(vec, prob):
    x = jnp.zeros(16)
    for k,v in prob.fixed_inputs.items(): x = x.at[_in(k)].set(v)
    for i,k in enumerate(prob.free_inputs): x = x.at[_in(k)].set(vec[i])
    return x

def loss_fn(vec64, fwd, prob):   # vec64 is float64 inside optimiser
    vec32 = vec64.astype(jnp.float32)
    y = fwd(assemble_x(vec32, prob))
    errs = [(y[_out(k)] - t)**2 for k,t in prob.target_outputs.items()]
    return jnp.sum(jnp.stack(errs)).astype(jnp.float64)

def _bounds_vec(prob, b):
    lo, hi = [], []
    for k in prob.free_inputs:
        l,h = b[k]; lo.append(l); hi.append(h)
    return jnp.array(lo, jnp.float64), jnp.array(hi, jnp.float64)

# -----------------------------------------------------------------------------
# Optimisers
# -----------------------------------------------------------------------------

def _adam_proj(fwd, prob, init64, lr, steps, lo=None, hi=None):
    proj = (lambda v: jnp.clip(v, lo, hi)) if lo is not None else (lambda v: v)
    vg   = jax.jit(jax.value_and_grad(lambda v: loss_fn(v, fwd, prob)))
    opt  = optax.adam(lr); state = opt.init(init64)
    def body(c,_):
        v,s = c; loss,g = vg(v); v = proj(optax.apply_updates(v, opt.update(g,s)[0])); return (v,s), loss
    (v,_), losses = jax.lax.scan(body, (init64,state), None, steps)
    return v.astype(jnp.float32), losses[-1]


def _lbfgs_unconstr(fwd, prob, init64, **kw):
    res = LBFGS(fun=lambda v: loss_fn(v,fwd,prob), implicit_diff=False, **kw).run(init64)
    return res.params.astype(jnp.float32), res.state.error


def _lbfgs_box(fwd, prob, init64, lo64, hi64, **kw):
    common = dict(fun=lambda v: loss_fn(v,fwd,prob), implicit_diff=False,
                  maxiter=kw.get("maxiter",200), tol=kw.get("tol",1e-9))
    # Try APIs --------------------------------------------------------------
    for args in (
        dict(lower=lo64, upper=hi64),
        dict(lower_bounds=lo64, upper_bounds=hi64),
        None,  # bounds via .run()
    ):
        try:
            solver = LBFGSB(**(args or {}), **common)
            res = solver.run(init64, bounds=(lo64,hi64)) if args is None else solver.run(init64)
            return res.params.astype(jnp.float32), res.state.error
        except TypeError:
            continue
    raise TypeError("LBFGSB signature not recognised; upgrade jaxopt â‰¥0.6")

# -----------------------------------------------------------------------------
# Public driver
# -----------------------------------------------------------------------------
class InverseSolver:
    def __init__(self, forward):
        self.fwd = forward

    def solve(self, prob: InverseProblem, *, init=None, bounds=None, method="lbfgs", **kw):
        if init is None: init = jnp.zeros(len(prob.free_inputs), jnp.float32)
        init64 = init.astype(jnp.float64)

        if bounds is not None:
            lo64, hi64 = _bounds_vec(prob, bounds)
            if method in ("lbfgs", "lbfgsb"):
                return _lbfgs_box(self.fwd, prob, init64, lo64, hi64, **kw)
            if method == "adam":
                return _adam_proj(self.fwd, prob, init64, kw.get('lr',1e-2), kw.get('n_steps',5000), lo64, hi64)
            raise ValueError("Bounds supported for adam / lbfgs[b] only.")

        # no bounds
        if method == "lbfgs":
            return _lbfgs_unconstr(self.fwd, prob, init64, **kw)
        if method == "adam":
            return _adam_proj(self.fwd, prob, init64, kw.get('lr',1e-2), kw.get('n_steps',500))
        raise ValueError(f"Unknown method {method!r}")