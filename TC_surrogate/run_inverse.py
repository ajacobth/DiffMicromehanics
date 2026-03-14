# run_inverse.py – TC surrogate inverse solver
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
import numpy as np
from absl import app, flags
from ml_collections import config_flags, ConfigDict

from inverse_model import (
    INPUT_FIELD_NAMES,
    OUTPUT_FIELD_NAMES,
    IN_IDX,
    OUT_IDX,
    create_predictor,
    InverseSolver,
    InverseProblem,
    assemble_x,
    make_orientation_sum_constraint,
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

_in  = lambda k: k if isinstance(k, int) else IN_IDX[k]
_out = lambda k: k if isinstance(k, int) else OUT_IDX[k]

# ---------------------------------------------------------------------------
# Helper – build InverseProblem
# ---------------------------------------------------------------------------

def _problem_from_cfg(inv: ConfigDict) -> InverseProblem:
    tf = lambda m: {k: float(v) for k, v in m.items()}
    free = [str(k) for k in inv.free_inputs]

    constraints = []
    c = make_orientation_sum_constraint(free)
    if c is not None:
        print("Constraint active: a11 + a22 <= 1.0")
        constraints.append(c)

    return InverseProblem(
        fixed_inputs=tf(inv.fixed_inputs),
        free_inputs=free,
        target_outputs=tf(inv.target_outputs),
        constraints=tuple(constraints),
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
    for name, idx in zip(INPUT_FIELD_NAMES, range(len(INPUT_FIELD_NAMES))):
        print(f"  {name:<20s}: {x_star[idx]:.6f}")

    print("\nOptimised free variables:")
    for k, v in opt_free.items():
        print(f"  {k:<20s}: {v:.6f}")

    if "a11" in opt_free and "a22" in opt_free:
        print(f"  {'a11 + a22':<20s}: {opt_free['a11'] + opt_free['a22']:.4f}  (constraint: <= 1.0)")

    print("\nPredicted outputs + % error:")
    for k, tgt in problem.target_outputs.items():
        pred = float(y_star[_out(k)])
        err  = 100.0 * abs(pred - tgt) / tgt if tgt != 0.0 else float('nan')
        print(f"  {k:<6s}: pred = {pred:.6f}   target = {tgt:.6f}   Percent_error = {err:+.4f}%")

    print(f"\nFinal loss: {float(final_loss):.4e}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
