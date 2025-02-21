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


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    dataset = get_dataset(config)
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()
    # Initialize model
    model = models.MICRO_SURROGATE(config)

    path = os.path.join(workdir, "ckpt", config.wandb.name)
    if os.path.exists(path):
        shutil.rmtree(path)
        
    # Initialize evaluator
    evaluator = models.MICRO_SURROGATE_Eval(config, model)
    
    mgr_options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3)
    ckpt_mgr = orbax.checkpoint.CheckpointManager(path, orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)
    
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
    input_dim = config.input_dim  # Number of input columns

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

        # Save checkpoint at the end of each epoch
        if config.saving.save_epoch == 0:
            save_checkpoint(model.state, path, ckpt_mgr)

    # Write summary time
    with open("time_summary.txt", "a") as f:
        f.write("\n"+ config.wandb.name+"--- %s seconds ---" % (time.time() - start_time_total))

    return model
