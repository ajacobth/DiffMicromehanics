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
        test_data =np.genfromtxt("validation_data.csv", delimiter=',', skip_header=1)  # Ensure dataset function loads test set

        # Get input and output sizes from config
        input_dim = config.input_dim
        output_dim = config.output_dim

        # Extract inputs and targets from test data
        test_inputs = test_data[:, :input_dim]
        test_targets = test_data[:, input_dim:]
        
        model =  models.MICRO_SURROGATE(config, input_mean=input_mean, input_std=input_std, 
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
    ckpt_mgr = orbax.checkpoint.CheckpointManager(
        abs_ckpt_path,
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        mgr_options
    )
    
    print("Waiting for JIT...")
    start_time_total = time.time()

    # Determine number of devices for pmap
    num_devices = jax.device_count()
    print(f"Using {num_devices} devices for parallel training.")

    # Ensure dataset is divisible by num_devices
    num_samples = dataset.shape[0]
    if num_samples % num_devices != 0:
        new_size = (num_samples // num_devices) * num_devices  # Round down to nearest multiple
        dataset = dataset[:new_size]  # Trim dataset to match devices
        print(f"Trimmed dataset to {new_size} samples for device alignment.")

    # Adjust batch size to be divisible by num_devices
    batch_size = config.training.batch_size
    batch_size_per_device = batch_size // num_devices
    if batch_size % num_devices != 0:
        print(f"Adjusting batch size: {batch_size} → {batch_size_per_device * num_devices} (divisible by {num_devices})")
        batch_size = batch_size_per_device * num_devices

    num_batches = num_samples // batch_size

    for epoch in range(config.training.max_epochs):
        start_time = time.time()
        
        # Shuffle dataset at the start of each epoch
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, num_samples)
        dataset_shuffled = dataset[perm]

        print(f"Epoch {epoch+1}/{config.training.max_epochs}, Num Batches: {num_batches}")

        for i in range(num_batches):
            batch_indices = jax.random.choice(
                jax.random.PRNGKey(epoch * num_batches + i),
                num_samples,
                (batch_size,),
                replace=False
            )
            batch_data = dataset_shuffled[batch_indices]

            # Split batch into inputs and targets
            batch_inputs = batch_data[:, :input_dim]
            batch_targets = batch_data[:, input_dim:]

            # Ensure batch is evenly split across devices
            batch_size_per_device = batch_size // num_devices
            
            # Reshape batch correctly for `pmap` (devices, batch_size_per_device, features)
            batch_inputs = jnp.reshape(batch_inputs, (num_devices, batch_size_per_device, -1))
            batch_targets = jnp.reshape(batch_targets, (num_devices, batch_size_per_device, -1))

            # Perform one step of training (pmap is already in model.step)
            model.state = model.step(model.state, batch_inputs, batch_targets)

            # Logging only on the main process
            if jax.process_index() == 0 and (i % config.logging.log_every_steps == 0):
                state = jax.device_get(jax.tree_map(lambda x: x[0], model.state))
                log_dict = evaluator(state, batch_inputs, batch_targets)
                wandb.log(log_dict, step=(epoch * num_batches + i))
                end_time = time.time()
                logger.log_iter(epoch, start_time, end_time, log_dict)

        # Save checkpoint at the end of each epoch
        if epoch % config.saving.save_epoch == 0:
            abs_ckpt_path = os.path.abspath(path)  # Convert path to absolute
            print(f"Saving checkpoint at: {abs_ckpt_path}")  # Debugging print
            save_checkpoint(model.state, abs_ckpt_path, ckpt_mgr)

    # Write summary time
    with open("time_summary.txt", "a") as f:
        f.write("\n"+ config.wandb.name+"--- %s seconds ---" % (time.time() - start_time_total))

    # Save normalization stats for future use (e.g., inference)
    jnp.savez(os.path.join(workdir, "normalization_stats.npz"), **norm_stats)

    return model
