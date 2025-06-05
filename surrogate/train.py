import os
import time
import shutil 

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb

from NN_surrogate.logging import Logger
from NN_surrogate.utils import save_checkpoint
import orbax 
import models
from utils import get_dataset
import numpy as np


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    dataset = get_dataset(config)
    wandb_config = config.wandb
    wandb.init(mode="online", project=wandb_config.project, name=wandb_config.name)

    # Compute mean and std for normalization
    input_dim = config.input_dim
    output_dim = config.output_dim

    inputs = dataset[:, :input_dim]
    targets = dataset[:, input_dim:]

    input_mean = jnp.mean(inputs, axis=0)
    input_std = jnp.std(inputs, axis=0) + 1e-8  # Prevent division by zero
    target_mean = jnp.mean(targets, axis=0)
    target_std = jnp.std(targets, axis=0) + 1e-8

    print(f"Normalizing dataset: Input mean {input_mean.shape}, Input std {input_std.shape}")
    print(f"Target mean {target_mean.shape}, Target std {target_std.shape}")

    # Normalize dataset
    inputs = (inputs - input_mean) / input_std
    targets = (targets - target_mean) / target_std
    dataset = jnp.concatenate([inputs, targets], axis=1)

    # Save normalization values for inference
    norm_stats = {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std
    }

    # Initialize logger
    logger = Logger()
    
    # Initialize model
    
    if config.use_train_val_test_split:

        # Load test dataset
        test_data =np.genfromtxt("composite_materials_validation_v3.csv", delimiter=',', skip_header=1)  # Ensure dataset function loads test set

        # Get input and output sizes from config
        input_dim = config.input_dim
        output_dim = config.output_dim

        # Extract inputs and targets from test data
        test_inputs = test_data[:, :input_dim]
        test_targets = test_data[:, input_dim:]
        
        if config.use_l2reg:
            
            model =  models.MICRO_SURROGATE_L2(config, input_mean=input_mean, input_std=input_std, 
                         output_mean=target_mean, output_std=target_std, 
                         x_val=test_inputs, y_val=test_targets)
        
        else:
            model =  models.MICRO_SURROGATE(config, input_mean=input_mean, input_std=input_std, 
                         output_mean=target_mean, output_std=target_std, 
                         x_val=test_inputs, y_val=test_targets)
        
    
    else:
        # Load test dataset
        test_data =np.genfromtxt("composite_materials_validation_v3.csv", delimiter=',', skip_header=1)  # Ensure dataset function loads test set

        # Get input and output sizes from config
        input_dim = config.input_dim
        output_dim = config.output_dim

        # Extract inputs and targets from test data
        test_inputs = test_data[:, :input_dim]
        test_targets = test_data[:, input_dim:]
        
        if config.use_l2reg:
            
            model =  models.MICRO_SURROGATE_L2(config, input_mean=input_mean, input_std=input_std, 
                         output_mean=target_mean, output_std=target_std, 
                         x_val=test_inputs, y_val=test_targets)
        
        else:
            model = models.MICRO_SURROGATE(config)

    path = os.path.join(workdir, "ckpt", config.wandb.name)

    # Ensure the checkpoint directory exists
    abs_ckpt_path = os.path.abspath(path)
    os.makedirs(abs_ckpt_path, exist_ok=True)

    # Initialize evaluator
    evaluator = models.MICRO_SURROGATE_Eval(config, model)

    # Initialize Checkpoint Manager with absolute path
    mgr_options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3)
    ckpt_dir = os.path.abspath(os.path.join(workdir, "ckpt", config.wandb.name))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_mgr = orbax.checkpoint.CheckpointManager(
        abs_ckpt_path,
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        mgr_options
    )
    
    print("Waiting for JIT...")
    start_time_total = time.time()

    num_devices = jax.device_count()
    print(f"Using {num_devices} device(s).")

    def trim_to_multiple(arr, multiple):
        excess = arr.shape[0] % multiple
        return arr if excess == 0 else arr[:-excess]

    # Align to devices first
    dataset = trim_to_multiple(dataset, num_devices)

    # Align to batch size *after* we possibly adjust batch size
    batch_size = config.training.batch_size
    if batch_size % num_devices != 0:
        batch_size = (batch_size // num_devices) * num_devices
        print(f"Adjusted batch size to {batch_size} for device divisibility.")

    dataset = trim_to_multiple(dataset, batch_size)   # ensures reshape works
    num_samples = dataset.shape[0]
    num_batches = num_samples // batch_size
    bs_per_dev = batch_size // num_devices

    # ------------------------------------------------------------------
    # 5.  Training loop
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(config.seed)
    start_wall = time.time()

    evaluator = models.MICRO_SURROGATE_Eval(config, model)

    for epoch in range(config.training.max_epochs):
        epoch_start = time.time()

        # shuffle once per epoch
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, num_samples)
        perm = perm.reshape((num_batches, batch_size))   # safe: exact multiple

        print(f"Epoch {epoch+1}/{config.training.max_epochs}")

        for i in range(num_batches):
            batch_idx   = perm[i]
            batch       = dataset[batch_idx]
            batch_inputs, batch_targets = (
                batch[:, :input_dim],
                batch[:,  input_dim:],
            )

            # reshape for pmap → (devices, bs_per_dev, features)
            batch_inputs  = batch_inputs.reshape(num_devices, bs_per_dev, input_dim)
            batch_targets = batch_targets.reshape(num_devices, bs_per_dev, output_dim)

            model.state = model.step(model.state, batch_inputs, batch_targets)

            global_step = epoch * num_batches + i
            if (jax.process_index() == 0
                    and global_step % config.logging.log_every_steps == 0):
                state_host = jax.device_get(tree_map(lambda x: x[0], model.state))
                log_dict   = evaluator(state_host, batch_inputs, batch_targets)
                wandb.log(log_dict, step=global_step)
                logger.log_iter(epoch, epoch_start, time.time(), log_dict)

        # checkpoint
        if epoch % config.saving.save_epoch == 0:
            print(f"Saving checkpoint → {ckpt_dir}")
            save_checkpoint(model.state, ckpt_dir, ckpt_mgr)

    # ------------------------------------------------------------------
    # 6.  Wrap-up
    # ------------------------------------------------------------------
    with open("time_summary.txt", "a") as f:
        f.write(f"\n{config.wandb.name} — {time.time() - start_wall:.1f} s")

    jnp.savez(os.path.join(workdir, "normalization_stats.npz"), **norm_stats)
    return model