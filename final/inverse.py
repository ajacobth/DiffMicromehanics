"""inverse.py – inverse design using the elastic surrogate.

The problem is fully described by a JSON file. See problem.json for the schema.

Usage
-----
    python inverse.py                          # uses problem.json in current dir
    python inverse.py --problem my_problem.json
    python inverse.py --problem p.json --model_dir models/elastic

The script prints the optimised free variables, predicted outputs, and %errors,
then writes results to <problem_stem>_result.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Mapping, NamedTuple, Optional, Sequence, Union

# X64 must be set before any JAX import
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
from jaxopt import LBFGS, LBFGSB

_HERE   = os.path.dirname(os.path.abspath(__file__))
_EL_DIR = os.path.normpath(os.path.join(_HERE, "..", "EL_surrogate"))
if _EL_DIR not in sys.path:
    sys.path.insert(0, _EL_DIR)

from forward import load_forward, MODELS_DIR

# ── types ─────────────────────────────────────────────────────────────────────
InputKey  = Union[int, str]
OutputKey = Union[int, str]

class InverseProblem(NamedTuple):
    fixed_inputs:   Mapping[str, float]
    free_inputs:    Sequence[str]
    target_outputs: Mapping[str, float]
    constraints:    Sequence[Callable] = ()


# ── constraints ───────────────────────────────────────────────────────────────
def make_orientation_sum_constraint(free_inputs: Sequence[str], limit: float = 1.0) -> Optional[Callable]:
    """Enforces a11 + a22 <= limit. Returns None if both aren't free."""
    names = list(free_inputs)
    if "a11" not in names or "a22" not in names:
        return None
    i11, i22 = names.index("a11"), names.index("a22")
    def c(vec):
        return vec[i11] + vec[i22] - limit
    return c


# ── core loss ─────────────────────────────────────────────────────────────────
def _assemble_x(vec, prob: InverseProblem, in_idx: dict, n_inputs: int) -> jnp.ndarray:
    x = jnp.zeros(n_inputs)
    for k, v in prob.fixed_inputs.items():
        x = x.at[in_idx[k]].set(v)
    for i, k in enumerate(prob.free_inputs):
        x = x.at[in_idx[k]].set(vec[i])
    return x


def _loss_fn(vec64, predict_array, prob: InverseProblem,
             in_idx: dict, out_idx: dict, n_inputs: int,
             penalty: float = 1e4) -> jnp.ndarray:
    vec32 = vec64.astype(jnp.float32)
    x     = _assemble_x(vec32, prob, in_idx, n_inputs)
    y     = predict_array(x)
    errs  = jnp.stack([(y[out_idx[k]] - t) ** 2 for k, t in prob.target_outputs.items()])
    loss  = jnp.sum(errs)
    for c in prob.constraints:
        loss = loss + penalty * jnp.maximum(0.0, c(vec64)) ** 2
    return loss.astype(jnp.float64)


# ── solvers ───────────────────────────────────────────────────────────────────
def _solve(predict_array, prob: InverseProblem, in_idx: dict, out_idx: dict,
           n_inputs: int, init64: jnp.ndarray,
           bounds: Optional[dict], method: str, penalty: float, **kw):

    loss = lambda v: _loss_fn(v, predict_array, prob, in_idx, out_idx, n_inputs, penalty)

    if bounds is not None:
        lo = jnp.array([bounds[k][0] for k in prob.free_inputs], jnp.float64)
        hi = jnp.array([bounds[k][1] for k in prob.free_inputs], jnp.float64)
    else:
        lo, hi = None, None

    if method == "adam":
        lr    = kw.get("lr", 1e-2)
        steps = int(kw.get("n_steps", 5000))
        proj  = (lambda v: jnp.clip(v, lo, hi)) if lo is not None else (lambda v: v)
        vg    = jax.jit(jax.value_and_grad(loss))
        opt   = optax.adam(lr)
        state = opt.init(init64)

        def body(carry, _):
            v, s = carry
            l, g = vg(v)
            v = proj(optax.apply_updates(v, opt.update(g, s)[0]))
            return (v, s), l

        (v, _), losses = jax.lax.scan(body, (init64, state), None, steps)
        return v.astype(jnp.float32), float(losses[-1])

    # lbfgs / lbfgsb
    maxiter = int(kw.get("maxiter", 300))
    tol     = float(kw.get("tol", 1e-9))

    if lo is not None:
        for api_kwargs in (
            dict(lower=lo, upper=hi),
            dict(lower_bounds=lo, upper_bounds=hi),
            None,  # pass bounds via .run()
        ):
            try:
                solver = LBFGSB(fun=loss, implicit_diff=False, maxiter=maxiter, tol=tol,
                                **(api_kwargs or {}))
                res = solver.run(init64, bounds=(lo, hi)) if api_kwargs is None else solver.run(init64)
                return res.params.astype(jnp.float32), float(res.state.error)
            except TypeError:
                continue
        raise TypeError("LBFGSB API not recognised; upgrade jaxopt >= 0.6")
    else:
        res = LBFGS(fun=loss, implicit_diff=False, maxiter=maxiter, tol=tol).run(init64)
        return res.params.astype(jnp.float32), float(res.state.error)


# ── main entry ────────────────────────────────────────────────────────────────
def run(problem_path: str, model_dir: Optional[str] = None) -> dict:
    with open(problem_path) as f:
        prob_dict = json.load(f)

    if model_dir is None:
        model_dir = os.path.join(MODELS_DIR, "elastic")

    # load model
    model_name = os.path.basename(os.path.normpath(model_dir))
    model      = load_forward(model_name)

    # validate that all requested fields exist in this model
    unknown_fixed  = [k for k in prob_dict["fixed_inputs"]  if k not in model.in_idx]
    unknown_free   = [k for k in prob_dict["free_inputs"]   if k not in model.in_idx]
    unknown_target = [k for k in prob_dict["target_outputs"] if k not in model.out_idx]
    if unknown_fixed:
        raise KeyError(f"Unknown fixed_inputs keys: {unknown_fixed}")
    if unknown_free:
        raise KeyError(f"Unknown free_inputs keys: {unknown_free}")
    if unknown_target:
        raise KeyError(f"Unknown target_outputs keys: {unknown_target}")

    fixed   = {k: float(v) for k, v in prob_dict["fixed_inputs"].items()}
    free    = list(prob_dict["free_inputs"])
    targets = {k: float(v) for k, v in prob_dict["target_outputs"].items()}
    bounds  = {k: tuple(v) for k, v in prob_dict.get("bounds", {}).items()} or None

    # build constraints
    constraints = []
    c = make_orientation_sum_constraint(free)
    if c is not None:
        print("Constraint active: a11 + a22 <= 1.0")
        constraints.append(c)

    prob = InverseProblem(
        fixed_inputs=fixed,
        free_inputs=free,
        target_outputs=targets,
        constraints=tuple(constraints),
    )

    solver_cfg = prob_dict.get("solver", {})
    method     = solver_cfg.get("method", "lbfgs")
    penalty    = float(solver_cfg.get("constraint_penalty", 1e4))

    # initial guess: use fixed_inputs values for free vars (or explicit override)
    if "init_free" in prob_dict:
        init = np.array(prob_dict["init_free"], dtype=np.float64)
    else:
        init = np.array([fixed[k] for k in free], dtype=np.float64)

    print(f"Solving inverse problem ({method}) with {len(free)} free variable(s)...")
    free_vec, final_err = _solve(
        model.predict_array, prob,
        model.in_idx, model.out_idx, len(model.input_fields),
        jnp.array(init, jnp.float64), bounds, method, penalty,
        maxiter=int(solver_cfg.get("maxiter", 300)),
        tol=float(solver_cfg.get("tol", 1e-9)),
        lr=float(solver_cfg.get("lr", 1e-2)),
        n_steps=int(solver_cfg.get("n_steps", 5000)),
    )

    # reconstruct full input + evaluate
    x_star = _assemble_x(free_vec, prob, model.in_idx, len(model.input_fields))
    y_star = model.predict_array(x_star)

    opt_free = {k: float(v) for k, v in zip(free, free_vec)}

    print("\n=== Inverse-design result ===")
    print("Optimised free variables:")
    for k, v in opt_free.items():
        print(f"  {k:<25s}: {v:.6f}")
    if "a11" in opt_free and "a22" in opt_free:
        s = opt_free["a11"] + opt_free["a22"]
        print(f"  {'a11 + a22':<25s}: {s:.6f}  (constraint: <= 1.0)")

    print(f"\nFinal optimiser error : {final_err:.4e}")
    print("\nPredicted outputs vs targets:")
    for k, tgt in targets.items():
        pred = float(y_star[model.out_idx[k]])
        err  = 100.0 * abs(pred - tgt) / tgt if tgt != 0.0 else float("nan")
        print(f"  {k:<8s}: pred = {pred:12.4f}   target = {tgt:12.4f}   err = {err:+.3f}%")

    result = {
        "free_variables":     opt_free,
        "predicted_outputs":  {k: float(y_star[model.out_idx[k]]) for k in model.output_fields},
        "target_outputs":     targets,
        "final_optimiser_error": final_err,
    }

    # write result JSON next to problem file
    stem        = os.path.splitext(problem_path)[0]
    result_path = stem + "_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to: {result_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverse design via elastic surrogate.")
    parser.add_argument("--problem",   default="problem.json",
                        help="Path to problem JSON file (default: problem.json)")
    parser.add_argument("--model_dir", default=None,
                        help="Path to model directory (default: models/elastic)")
    args = parser.parse_args()
    run(args.problem, args.model_dir)
