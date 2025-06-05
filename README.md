# DiffMicromehanics Surrogate Models

This repository provides neural network surrogates for predicting the effective elastic properties of composite materials. The `surrogate/` folder contains the training utilities, model definitions and example configuration files used to reproduce the results.

## Training the Models

All training is driven by configuration files located under `surrogate/configs/`. Each file defines the network architecture, optimiser settings and dataset options. To start a run, supply the chosen configuration together with a working directory to store checkpoints and normalisation statistics.

Run the standard multilayer perceptron (without L2 regularisation):

```bash
python surrogate/main.py --config surrogate/configs/case_1.py --workdir <path>
```

Run a model with L2 regularisation enabled (see `case_2.py`):

```bash
python surrogate/main.py --config surrogate/configs/case_2.py --workdir <path>
```

A Bayesian neural network variant can be trained with `train_bnn.py` and the `case_bnn.py` configuration:

```bash
python surrogate/train_bnn.py --config surrogate/configs/case_bnn.py --workdir <path>
```

Checkpoints are written under `workdir/ckpt/<run_name>` and normalisation statistics are stored in `workdir/normalization_stats.npz`.

## Running the Inverse Design Solver

The script `surrogate/run_inverse.py` solves an optimisation problem where selected input variables are adjusted so that the surrogate output matches user specified targets. The problem definition is read from the `inverse` section of the chosen configuration file.

Example usage (using the inverse block from `case_2.py`):

```bash
python surrogate/run_inverse.py --config surrogate/configs/case_2.py --workdir <path>
```

The solver supports `adam`, `lbfgs` and `lbfgsb` (bounded L‑BFGS). Use `--optim` to override the optimiser defined in the config:

```bash
python surrogate/run_inverse.py --config surrogate/configs/case_2.py --workdir <path> --optim lbfgsb
```

After optimisation the script prints the full input vector, the values found for each free variable and the predicted outputs alongside percentage errors with respect to the targets.

