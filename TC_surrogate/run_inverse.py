# run_inverse.py – now supports bounds + LBFGSB
# ---------------------------------------------------------------------------
# Loads surrogate + config, converts `config.inverse` to InverseProblem, builds
# initial guess, *optionally reads `bounds`* (dict of name → (lo,hi)), and runs
# one of {adam, lbfgs, lbfgsb}.  Prints full x*, free-var dict, outputs, loss.
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = 'cpu'   # ▸ must appear before any JAX import
import jax
jax.config.update("jax_enable_x64", True)
# Deterministic
import numpy as np
from absl import app, flags
from ml_collections import config_flags, ConfigDict
from typing import Any, Mapping

from inverse_model import (
    create_predictor,
    InverseSolver,
    InverseProblem,
    assemble_x,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "workdir", default=".",
    help="Directory containing checkpoint & normalization_stats.npz",
)

config_flags.DEFINE_config_file(
    name="config", default="./configs/default.py", lock_config=True,
)

flags.DEFINE_enum(
    "optim", default=None,
    enum_values=["lbfgs", "lbfgsb", "adam"],
    help="Override optimiser (else use config.inverse.optim).",
)

INPUT_FIELD_NAMES = [
    "fiber_e1", "fiber_e2", "fiber_g12", "fiber_nu12",
    "fiber_nu23", "fiber_aspect", "fiber_massfrac", "fiber_density",
    "matrix_modulus", "matrix_poissonratio", "matrix_density",
    "a11", "a22", "a12", "a13", "a23",
]
OUTPUT_FIELD_NAMES = [
    "E1", "E2", "E3", "nu12", "nu13",
    "G12", "G13", "G23", "nu23",
]
IN_IDX  = {n: i for i, n in enumerate(INPUT_FIELD_NAMES)}
OUT_IDX = {n: i for i, n in enumerate(OUTPUT_FIELD_NAMES)}

_in  = lambda k: k if isinstance(k,int) else IN_IDX[k]
_out = lambda k: k if isinstance(k,int) else OUT_IDX[k]

# ---------------------------------------------------------------------------
# Helper – build InverseProblem
# ---------------------------------------------------------------------------

def _problem_from_cfg(inv: ConfigDict) -> InverseProblem:
    tf = lambda m: {k: float(v) for k, v in m.items()}
    return InverseProblem(
        fixed_inputs=tf(inv.fixed_inputs),
        free_inputs=[str(k) for k in inv.free_inputs],
        target_outputs=tf(inv.target_outputs),
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_argv):
    cfg: ConfigDict = FLAGS.config
    workdir = FLAGS.workdir

    if "inverse" not in cfg:
        raise ValueError("Config missing `inverse` section.")

    inv_cfg = cfg.inverse
    problem = _problem_from_cfg(inv_cfg)

    forward = create_predictor(cfg, workdir)
    solver  = InverseSolver(forward)

    # choose optimiser -------------------------------------------------------
    optim_method = FLAGS.optim or inv_cfg.get("optim", "lbfgs")

    solver_kwargs = {}
    if optim_method in ("adam",):
        solver_kwargs.update(lr=float(inv_cfg.get("adam_lr", 1e-2)),
                             n_steps=int(inv_cfg.get("adam_steps", 500)))
    else:   # lbfgs / lbfgsb
        solver_kwargs.update(maxiter=int(inv_cfg.get("lbfgs_maxiter", 300)),
                             tol=float(inv_cfg.get("lbfgs_tol", 1e-9)))

    # initial guess ----------------------------------------------------------
    if "init_free" in inv_cfg:
        init_guess = np.asarray(inv_cfg.init_free, np.float32)
    else:
        init_guess = np.asarray([problem.fixed_inputs[k]
                                 for k in problem.free_inputs], np.float32)

    # bounds (optional) ------------------------------------------------------
    bounds_dict = inv_cfg.get("bounds", None)

    # solve ------------------------------------------------------------------
    free_vec, final_loss = solver.solve(
        problem,
        init=init_guess,
        method=optim_method,
        bounds=bounds_dict,
        **solver_kwargs,
    )

    # report -----------------------------------------------------------------
    x_star = assemble_x(free_vec, problem)
    y_star = forward(x_star)
    opt_free = {k: float(v) for k, v in zip(problem.free_inputs, free_vec)}

    print("\n=== Inverse-design result ===")
    print("Optimised full input vector:")
    for name, idx in zip(INPUT_FIELD_NAMES, range(16)):
        print(f"  {name:<20s}: {x_star[idx]:.4f}")
    
    print("\nOptimised free variables:")
    for k, v in opt_free.items():
        print(f"  {k:<20s}: {v:.4f}")
    
    print("\nPredicted outputs + % error:")
    pct_err = {}
    for k, tgt in problem.target_outputs.items():
        pred = float(y_star[_out(k)])
        err  = 100.0 * abs(pred - tgt) / tgt
        pct_err[k] = err
        print(f"  {k:<5s}: pred = {pred:.4f}   target = {tgt:.4f}   Percent_error = {err:+6.4f}%")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
