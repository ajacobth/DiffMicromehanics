# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`THEL_surrogate` is a JAX/Flax neural network surrogate for predicting the **thermoelastic** effective properties of fiber-reinforced composite materials. It predicts 15 outputs (9 elastic moduli + 6 CTE components) from 19 microstructure/material inputs. Sister directories (`EL_surrogate`, `TC_surrogate`) follow identical patterns for elastic and thermal conductivity surrogates.

## Commands

**Train:**
```bash
python main.py --config configs/case_4.py --workdir .
```

**Evaluate (generates plots + metrics.txt in `eval_figs/<run_name>/`):**
```bash
# Set config.mode = "eval" in the config file, then:
python main.py --config configs/case_4.py --workdir .
```

**Inverse design:**
```bash
python run_inverse.py --config configs/case_4.py --workdir .
# Override optimizer at the CLI:
python run_inverse.py --config configs/case_4.py --workdir . --optim lbfgsb
```

**GUI prediction:**
```bash
# Set config.mode = "gui" in the config file, then:
python main.py --config configs/case_4.py --workdir .
```

## Architecture

### Two-level model hierarchy

`NN_surrogate/` is the **generic base library** (not domain-specific):
- `archs.py` — Flax `nn.Module` definitions: `Mlp`, `BatchNorm_Mlp`, `BayesianMlp`
- `models.py` — `SURROGATE` base class: handles `TrainState`, optimizer, `pmap`-based `step()`, loss weighting via `grad_norm`
- `evaluator.py` — `BaseEvaluator`: logs losses, weights, grad norms to a dict
- `utils.py` / `logging.py` — checkpoint save/restore, training logger

`models.py` (top-level) is the **domain-specific layer** that subclasses `SURROGATE`:
- `MICRO_SURROGATE` — MSE-only loss
- `MICRO_SURROGATE_L2` — MSE + L2 weight regularization (biases excluded, 1D arrays skipped)
- `MICRO_SURROGATE_BNN` — Bayesian variational inference (NLL + KL loss)
- `MICRO_SURROGATE_Eval` — subclasses `BaseEvaluator`, adds validation MSE logging

### Data flow

1. `utils.py:get_dataset()` loads `THELPSD_dataset_train_V2.csv` (columns: 19 inputs then 15 outputs)
2. `train.py` computes z-score normalization stats on train set, saves to `normalization_stats.npz`
3. Training uses `pmap` across devices; last batch is padded with wrap-around indices (no dropping)
4. Checkpoints written by `orbax` under `ckpt/<run_name>/`; `eval.py` restores and denormalizes predictions

### Config files (`configs/case_*.py`)

Each config returns a `ml_collections.ConfigDict` with sections:
- `arch`: `arch_name`, `hidden_dim` (tuple), `out_dim`, `activation`
- `training`: `max_epochs`, `batch_size`
- `optim`: Adam with exponential LR decay
- `weighting`: `grad_norm` scheme with `init_weights` dict (keys must match loss names returned by `losses()`)
- `logging` / `saving`: log cadence, checkpoint interval
- `use_l2reg`: switches between `MICRO_SURROGATE` vs `MICRO_SURROGATE_L2`
- `inverse`: `fixed_inputs`, `free_inputs`, `target_outputs`, `bounds`, optimizer settings

### Inverse design (`inverse_model.py` + `run_inverse.py`)

`run_inverse.py` loads the surrogate, reads the `config.inverse` block, and optimizes `free_inputs` to match `target_outputs` while `fixed_inputs` remain constant. Supports `adam` (via optax), `lbfgs`, and `lbfgsb` (bounded). `INPUT_FIELD_NAMES` in `run_inverse.py` must match the order in the training CSV.

## Key conventions

- **Dataset files** must be present in the working directory at runtime: `THELPSD_dataset_train_V2.csv`, `THELPSD_dataset_validate_V2.csv`, `THELPSD_dataset_test_V2.csv` (19 input + 15 output columns, with header row).
- **Loss weight keys** in `weighting.init_weights` must exactly match the dict keys returned by the model's `losses()` method (e.g., `{"mse": 1., "l2": 1e-6}` for `MICRO_SURROGATE_L2`).
- Training logs wall time to `time_summary.txt` (appended, not overwritten).
- `eval.py` expects `normalization_stats.npz` in `workdir` — run training before evaluation.
- `run_inverse.py` forces `JAX_PLATFORM_NAME=cpu` and `jax_enable_x64=True` at the top of the file.
