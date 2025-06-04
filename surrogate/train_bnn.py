import os
import time
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


def train_and_evaluate_bnn(config: ml_collections.ConfigDict, workdir: str):
    dataset = get_dataset(config)
    wandb_config = config.wandb
    wandb.init(mode="online", project=wandb_config.project, name=wandb_config.name)

    input_dim = config.input_dim
    output_dim = config.output_dim

    inputs = dataset[:, :input_dim]
    targets = dataset[:, input_dim:]

    input_mean = jnp.mean(inputs, axis=0)
    input_std = jnp.std(inputs, axis=0) + 1e-8
    target_mean = jnp.mean(targets, axis=0)
    target_std = jnp.std(targets, axis=0) + 1e-8

    inputs = (inputs - input_mean) / input_std
    targets = (targets - target_mean) / target_std
    dataset = jnp.concatenate([inputs, targets], axis=1)

    norm_stats = {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }

    logger = Logger()

    dataset_size = dataset.shape[0]

    if config.use_train_val_test_split:
        test_data = np.genfromtxt("validation_data.csv", delimiter=",", skip_header=1)
        test_inputs = test_data[:, :input_dim]
        test_targets = test_data[:, input_dim:]
        model = models.MICRO_SURROGATE_BNN(
            config,
            dataset_size,
            input_mean=input_mean,
            input_std=input_std,
            output_mean=target_mean,
            output_std=target_std,
            x_val=test_inputs,
            y_val=test_targets,
        )
    else:
        model = models.MICRO_SURROGATE_BNN(config, dataset_size)

    path = os.path.join(workdir, "ckpt", config.wandb.name)
    abs_ckpt_path = os.path.abspath(path)
    os.makedirs(abs_ckpt_path, exist_ok=True)

    evaluator = models.MICRO_SURROGATE_Eval(config, model)

    mgr_options = orbax.checkpoint.CheckpointManagerOptions(save_interval_steps=1, max_to_keep=3)
    ckpt_mgr = orbax.checkpoint.CheckpointManager(
        abs_ckpt_path,
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        mgr_options,
    )

    start_time_total = time.time()
    num_devices = jax.device_count()
    num_samples = dataset.shape[0]
    if num_samples % num_devices != 0:
        new_size = (num_samples // num_devices) * num_devices
        dataset = dataset[:new_size]
    batch_size = config.training.batch_size
    batch_size_per_device = batch_size // num_devices
    if batch_size % num_devices != 0:
        batch_size = batch_size_per_device * num_devices
    num_batches = num_samples // batch_size

    for epoch in range(config.training.max_epochs):
        start_time = time.time()
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, num_samples)
        dataset_shuffled = dataset[perm]

        for i in range(num_batches):
            batch_key = jax.random.PRNGKey(epoch * num_batches + i)
            batch_indices = jax.random.choice(batch_key, num_samples, (batch_size,), replace=False)
            batch_data = dataset_shuffled[batch_indices]

            batch_inputs = batch_data[:, :input_dim]
            batch_targets = batch_data[:, input_dim:]

            batch_inputs = jnp.reshape(batch_inputs, (num_devices, batch_size_per_device, -1))
            batch_targets = jnp.reshape(batch_targets, (num_devices, batch_size_per_device, -1))
            rngs = jax.random.split(batch_key, num_devices)

            model.state = model.step(model.state, batch_inputs, batch_targets, rngs)

            if jax.process_index() == 0 and (i % config.logging.log_every_steps == 0):
                state = jax.device_get(jax.tree_map(lambda x: x[0], model.state))
                log_dict = evaluator(state, batch_inputs, batch_targets, rngs[0])
                wandb.log(log_dict, step=(epoch * num_batches + i))
                end_time = time.time()
                logger.log_iter(epoch, start_time, end_time, log_dict)

        if epoch % config.saving.save_epoch == 0:
            save_checkpoint(model.state, abs_ckpt_path, ckpt_mgr)

    with open("time_summary.txt", "a") as f:
        f.write("\n" + config.wandb.name + "--- %s seconds ---" % (time.time() - start_time_total))

    jnp.savez(os.path.join(workdir, "normalization_stats.npz"), **norm_stats)

    return model
