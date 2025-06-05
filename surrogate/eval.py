"""
Created on Wed Jul 10 17:10:00 2024

@author: akshayjacobthomas
"""

import os

import ml_collections
from jax import vmap
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from NN_surrogate.utils import restore_checkpoint
import models
from utils import get_dataset
#from get_solution import get_solution, get_solution_plot

import contextlib
plt.rcParams['text.usetex'] = True

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    
    # Restore model
    model = models.MICRO_SURROGATE_L2(config)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    state = restore_checkpoint(model.state, ckpt_path)
    params = state['params']  # Extract trained model parameters
    
    # Load test dataset
    test_data =np.genfromtxt("composite_materials_test_v3.csv", delimiter=',', skip_header=1)  # Ensure dataset function loads test set

    # Get input and output sizes from config
    input_dim = config.input_dim
    output_dim = config.output_dim

    # Extract inputs and targets from test data
    test_inputs = test_data[:, :input_dim]
    test_targets = test_data[:, input_dim:]

    # Load normalization statistics
    norm_stats_path = os.path.join(workdir, "normalization_stats.npz")
    norm_stats = np.load(norm_stats_path)

    input_mean = jnp.array(norm_stats["input_mean"])
    input_std = jnp.array(norm_stats["input_std"])
    target_mean = jnp.array(norm_stats["target_mean"])
    target_std = jnp.array(norm_stats["target_std"])

    # Normalize test inputs using stored stats
    test_inputs = (test_inputs - input_mean) / input_std

    # Model predictions
    test_preds = model.u_net(params, test_inputs)

    # Denormalize predictions to original scale
    test_preds = (test_preds * target_std) + target_mean

    # Compute error metrics
    mse = jnp.mean((test_preds - test_targets) ** 2)  # Mean Squared Error
    rmse = jnp.sqrt(mse)  # Root Mean Squared Error
    mae = jnp.mean(jnp.abs(test_preds - test_targets))  # Mean Absolute Error
    r2 = 1 - jnp.sum((test_preds - test_targets) ** 2) / jnp.sum((test_targets - jnp.mean(test_targets)) ** 2)  # R² Score

    # Print evaluation results
    print(f"Evaluation Results:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R² Score: {r2:.6f}")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
