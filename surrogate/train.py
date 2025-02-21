

#y (bead width direction)
#^
#|
#|
#|
#|-------------->x deposition direction
#0

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
    evaluator = models.MICRO_SURROGATE(config, model)
    
    mgr_options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3)
    ckpt_mgr = orbax.checkpoint.CheckpointManager(path, orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()), mgr_options)
    
    print("Waiting for JIT...")
    start_time_total = time.time()

    # Prepare data batching
    batch_size = config.training.batch_size
    num_samples = dataset.shape[0]  # Assume dataset is a 2D array
    num_batches = num_samples // batch_size

    input_dim = config.input_dim  # Number of input columns

    for epoch in range(config.training.max_epochs):
        start_time = time.time()
        
        # Shuffle dataset at the start of each epoch
        perm = jax.random.permutation(jax.random.PRNGKey(epoch), num_samples)
        dataset_shuffled = dataset[perm]

        for i in range(num_batches):
            batch_data = dataset_shuffled[i * batch_size: (i + 1) * batch_size]

            # Split batch into inputs and targets
            batch_inputs = batch_data[:, :input_dim]  # First n columns are inputs
            batch_targets = batch_data[:, input_dim:]  # Remaining columns are targets

            batch = {"inputs": batch_inputs, "targets": batch_targets}
            
            # Perform one step of training
            model.state = model.step(model.state, batch)

            # Log training metrics, only use host 0 to record results
            if jax.process_index() == 0 and (i % config.logging.log_every_steps == 0):
                # Get the first replica of the state and batch
                state = jax.device_get(jax.tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(jax.tree_map(lambda x: x[0], batch))
                
                log_dict = evaluator(state, batch)
                wandb.log(log_dict, step=(epoch * num_batches + i))

                end_time = time.time()
                logger.log_iter(epoch * num_batches + i, start_time, end_time, log_dict)

            # Saving
            if config.saving.save_every_steps is not None and ((epoch * num_batches + i + 1) % config.saving.save_every_steps == 0):
                save_checkpoint(model.state, path, ckpt_mgr)

    # Write summary time
    with open("time_summary.txt", "a") as f:
        f.write("\n"+ config.wandb.name+"--- %s seconds ---" % (time.time() - start_time_total))

    return model