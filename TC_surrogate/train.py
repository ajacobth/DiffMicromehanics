import os
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import wandb
import orbax
import ml_collections

from NN_surrogate.logging import Logger
from NN_surrogate.utils import save_checkpoint
import models
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # ------------------------------------------------------------
    # 0) W&B init
    # ------------------------------------------------------------
    dataset = get_dataset(config)  # expected shape (N, input_dim + output_dim)
    wandb_config = config.wandb
    wandb.init(mode="online", project=wandb_config.project, name=wandb_config.name)

    input_dim = int(config.input_dim)
    output_dim = int(config.output_dim)

    # ------------------------------------------------------------
    # 1) Compute normalization stats on TRAIN dataset
    # ------------------------------------------------------------
    inputs = dataset[:, :input_dim]
    targets = dataset[:, input_dim:]

    input_mean = jnp.mean(inputs, axis=0)
    input_std = jnp.std(inputs, axis=0) + 1e-8
    target_mean = jnp.mean(targets, axis=0)
    target_std = jnp.std(targets, axis=0) + 1e-8

    print(f"Normalizing dataset: input_mean {input_mean.shape}, input_std {input_std.shape}")
    print(f"                    target_mean {target_mean.shape}, target_std {target_std.shape}")

    # Normalize training dataset
    inputs = (inputs - input_mean) / input_std
    targets = (targets - target_mean) / target_std
    dataset = jnp.concatenate([inputs, targets], axis=1)

    norm_stats = {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }

    # ------------------------------------------------------------
    # 2) Logger
    # ------------------------------------------------------------
    logger = Logger()

    # ------------------------------------------------------------
    # 3) Load validation/test set (raw). Model can normalize internally.
    # ------------------------------------------------------------
    test_data = np.genfromtxt("TCPSD_dataset_validate.csv", delimiter=",", skip_header=1)
    test_inputs = jnp.asarray(test_data[:, :input_dim])
    test_targets = jnp.asarray(test_data[:, input_dim:])

    # ------------------------------------------------------------
    # 4) Initialize model
    # ------------------------------------------------------------
    if config.use_l2reg:
        model = models.MICRO_SURROGATE_L2(
            config,
            input_mean=input_mean,
            input_std=input_std,
            output_mean=target_mean,
            output_std=target_std,
            x_val=test_inputs,
            y_val=test_targets,
        )
    else:
        model = models.MICRO_SURROGATE(
            config,
            input_mean=input_mean,
            input_std=input_std,
            output_mean=target_mean,
            output_std=target_std,
            x_val=test_inputs,
            y_val=test_targets,
        )

    evaluator = models.MICRO_SURROGATE_Eval(config, model)

    # ------------------------------------------------------------
    # 5) Checkpoint manager
    # ------------------------------------------------------------
    ckpt_dir = os.path.abspath(os.path.join(workdir, "ckpt", config.wandb.name))
    os.makedirs(ckpt_dir, exist_ok=True)

    mgr_options = orbax.checkpoint.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=int(config.saving.num_keep_ckpts) if "num_keep_ckpts" in config.saving else 3,
    )
    ckpt_mgr = orbax.checkpoint.CheckpointManager(
        ckpt_dir,
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
        mgr_options,
    )

    # ------------------------------------------------------------
    # 6) Devices + batch sizing (NO DROPPING)
    # ------------------------------------------------------------
    num_devices = jax.device_count()
    print(f"Using {num_devices} device(s).")

    # Make batch_size divisible by num_devices for pmap reshape
    batch_size = int(config.training.batch_size)
    if batch_size % num_devices != 0:
        batch_size = (batch_size // num_devices) * num_devices
        if batch_size == 0:
            raise ValueError(f"batch_size too small for {num_devices} devices.")
        print(f"Adjusted batch_size → {batch_size} to be divisible by num_devices.")

    bs_per_dev = batch_size // num_devices

    num_samples = int(dataset.shape[0])
    num_batches = int(np.ceil(num_samples / batch_size))

    print(f"num_samples={num_samples}, batch_size={batch_size}, num_batches={num_batches}, "
          f"total_steps={int(config.training.max_epochs) * num_batches}")
    print(f"steps_per_epoch={num_batches}")

    # ------------------------------------------------------------
    # 7) Training loop
    #     - no dropping: last batch wraps indices from start of perm
    #     - logging: once per epoch (NOT global step)
    # ------------------------------------------------------------
    rng = jax.random.PRNGKey(int(config.seed))
    start_wall = time.time()

    print("Waiting for JIT...")

    for epoch in range(int(config.training.max_epochs)):
        epoch_start = time.time()

        # Shuffle all samples once per epoch
        rng, perm_key = jax.random.split(rng)
        perm = jax.random.permutation(perm_key, num_samples)

        if epoch % int(config.logging.print_every_epochs) == 0:
            print(f"\nEpoch {epoch+1}/{config.training.max_epochs}")

        # We'll keep the last batch around for end-of-epoch train logging
        last_batch_inputs = None
        last_batch_targets = None

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)

            batch_idx = perm[start:end]
            current_bs = int(batch_idx.shape[0])

            # Wrap-around to fill to full batch_size (no zeros, no dropping)
            if current_bs < batch_size:
                pad_needed = batch_size - current_bs
                pad_idx = perm[:pad_needed]
                batch_idx = jnp.concatenate([batch_idx, pad_idx], axis=0)

            batch = dataset[batch_idx]

            batch_inputs = batch[:, :input_dim]
            batch_targets = batch[:, input_dim:]

            # Reshape for pmap: (devices, bs_per_dev, features)
            batch_inputs = batch_inputs.reshape(num_devices, bs_per_dev, input_dim)
            batch_targets = batch_targets.reshape(num_devices, bs_per_dev, output_dim)

            model.state = model.step(model.state, batch_inputs, batch_targets)

            last_batch_inputs = batch_inputs
            last_batch_targets = batch_targets

        # --------------------------------------------------
        # End-of-epoch logging (once per epoch)
        # --------------------------------------------------
        if jax.process_index() == 0:
            # Bring one replica of state to host for eval/logging
            state_host = jax.device_get(tree_map(lambda x: x[0], model.state))

            # Evaluate on the last train batch (cheap, stable)
            log_dict = evaluator(state_host, last_batch_inputs, last_batch_targets)

            # Namespace train metrics (optional but recommended)
            log_dict = {f"train/{k}": v for k, v in log_dict.items()}

            # If your evaluator/model also injects validation metrics internally,
            # they will appear as additional keys in log_dict.
            wandb.log(log_dict, step=epoch)
            logger.log_iter(epoch, epoch_start, time.time(), log_dict)

        # --------------------------------------------------
        # Checkpoint every save_epoch epochs (treat as interval)
        # --------------------------------------------------
        save_every = int(config.saving.save_epoch)
        if save_every > 0 and ((epoch + 1) % save_every == 0):
            print(f"Saving checkpoint → {ckpt_dir}")
            save_checkpoint(model.state, ckpt_dir, ckpt_mgr)

    # ------------------------------------------------------------
    # 8) Wrap-up
    # ------------------------------------------------------------
    elapsed = time.time() - start_wall
    with open("time_summary.txt", "a") as f:
        f.write(f"\n{config.wandb.name} — {elapsed:.1f} s")

    jnp.savez(os.path.join(workdir, "normalization_stats.npz"), **norm_stats)
    return model
