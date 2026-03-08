"""forward.py – unified forward evaluator for elastic and thermoelastic surrogates.

Drop-in workflow
----------------
After training, run export_model.py in the training folder. It will populate:
    models/{elastic,thermoelastic}/
        model_config.json
        normalization_stats.npz
        ckpt/<checkpoint_name>/

Usage
-----
    from forward import load_forward

    fwd, meta = load_forward("elastic")
    outputs = fwd({"e1": 240e3, "e2": 15e3, ..., "a11": 0.6, ...})
    print(outputs["E1"])

    # or thermoelastic
    fwd_th, meta_th = load_forward("thermoelastic")
    outputs_th = fwd_th({...})
"""
from __future__ import annotations

import json
import os
import sys
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# ── shared NN_surrogate code lives in EL_surrogate ──────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_EL_DIR = os.path.normpath(os.path.join(_HERE, "..", "EL_surrogate"))
if _EL_DIR not in sys.path:
    sys.path.insert(0, _EL_DIR)

import ml_collections
from NN_surrogate.utils import restore_checkpoint
import models as _model_module

MODELS_DIR = os.path.join(_HERE, "models")


# ── public types ─────────────────────────────────────────────────────────────
class ForwardModel(NamedTuple):
    """Container returned by load_forward."""
    predict:       Callable   # dict[str, float] -> dict[str, float]  (user-facing)
    predict_array: Callable   # jnp.ndarray -> jnp.ndarray            (for optimizer)
    input_fields:  list
    output_fields: list
    in_idx:        dict       # field_name -> position in input vector
    out_idx:       dict       # field_name -> position in output vector
    output_mean:   jnp.ndarray  # mu_out used to normalise targets for the loss
    output_std:    jnp.ndarray  # sig_out used to normalise targets for the loss


# ── internal helpers ─────────────────────────────────────────────────────────
def make_ml_config(cfg: dict) -> ml_collections.ConfigDict:
    """Convert a model_config.json dict into an ml_collections.ConfigDict."""
    c = ml_collections.ConfigDict()
    c.arch            = ml_collections.ConfigDict()
    c.arch.arch_name  = cfg["arch_name"]
    c.arch.hidden_dim = tuple(cfg["hidden_dim"])
    c.arch.out_dim    = len(cfg["output_fields"])
    c.arch.activation = cfg["activation"]
    c.input_dim       = len(cfg["input_fields"])
    c.output_dim      = len(cfg["output_fields"])
    c.wandb           = ml_collections.ConfigDict()
    c.wandb.name      = cfg["checkpoint_name"]
    c.use_l2reg       = cfg.get("use_l2reg", True)
    c.seed            = 0

    # Required by _create_train_state / _create_optimizer (values unused at inference)
    c.optim = ml_collections.ConfigDict()
    c.optim.optimizer        = "Adam"
    c.optim.beta1            = 0.9
    c.optim.beta2            = 0.999
    c.optim.eps              = 1e-8
    c.optim.learning_rate    = 1e-3
    c.optim.decay_rate       = 0.9
    c.optim.decay_steps      = 1e6
    c.optim.grad_accum_steps = 0

    init_weights = {"mse": 1.0, "l2": 1e-6} if c.use_l2reg else {"mse": 1.0}
    c.weighting = ml_collections.ConfigDict()
    c.weighting.scheme             = "grad_norm"
    c.weighting.init_weights       = ml_collections.ConfigDict(init_weights)
    c.weighting.momentum           = 0.9
    c.weighting.update_every_steps = 1e12

    return c


# ── public API ───────────────────────────────────────────────────────────────
def load_forward(model_name: str) -> ForwardModel:
    """Load a surrogate model from models/<model_name>/ and return a ForwardModel.

    Parameters
    ----------
    model_name : "elastic" | "thermoelastic"

    Returns
    -------
    ForwardModel namedtuple with:
        .predict(inputs_dict)  -> outputs_dict    (dict-in / dict-out)
        .predict_array(x)      -> y               (jnp array in / out, JIT-compiled)
        .input_fields, .output_fields, .in_idx, .out_idx
    """
    model_dir = os.path.join(MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Run export_model.py in the training folder first."
        )

    with open(os.path.join(model_dir, "model_config.json")) as f:
        cfg_dict = json.load(f)

    ml_cfg    = make_ml_config(cfg_dict)
    surrogate = _model_module.MICRO_SURROGATE_L2(ml_cfg)

    ckpt_dir = os.path.join(model_dir, "ckpt", cfg_dict["checkpoint_name"])
    params   = restore_checkpoint(surrogate.state, ckpt_dir)["params"]

    stats   = np.load(os.path.join(model_dir, "normalization_stats.npz"))
    mu_in   = jnp.array(stats["input_mean"],  dtype=jnp.float32)
    sig_in  = jnp.array(stats["input_std"],   dtype=jnp.float32)
    mu_out  = jnp.array(stats["target_mean"], dtype=jnp.float32)
    sig_out = jnp.array(stats["target_std"],  dtype=jnp.float32)

    input_fields  = cfg_dict["input_fields"]
    output_fields = cfg_dict["output_fields"]
    in_idx  = {n: i for i, n in enumerate(input_fields)}
    out_idx = {n: i for i, n in enumerate(output_fields)}

    @jax.jit
    def predict_array(x: jnp.ndarray) -> jnp.ndarray:
        """Raw array forward pass. Input shape: (n_inputs,). Output: (n_outputs,)."""
        xn = (x[None] - mu_in) / sig_in
        yn = surrogate.u_net(params, xn)
        return (yn * sig_out + mu_out)[0]

    def predict(inputs_dict: dict) -> dict:
        """Dict-in / dict-out forward pass. All input keys must be provided."""
        missing = [k for k in input_fields if k not in inputs_dict]
        if missing:
            raise KeyError(f"Missing input keys: {missing}")
        x = jnp.array([float(inputs_dict[k]) for k in input_fields], dtype=jnp.float32)
        y = predict_array(x)
        return {k: float(v) for k, v in zip(output_fields, y)}

    return ForwardModel(
        predict=predict,
        predict_array=predict_array,
        input_fields=input_fields,
        output_fields=output_fields,
        in_idx=in_idx,
        out_idx=out_idx,
        output_mean=mu_out,
        output_std=sig_out,
    )
