import os

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, grad, tree_map
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

from flax.training import checkpoints
import orbax
import flax 

# definging things for checkpoint




def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]


@partial(jit, static_argnums=(0,))
def jacobian_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = grad(apply_fn, argnums=0)(params, *args)
    J, _ = ravel_pytree(J)
    return J


@partial(jit, static_argnums=(0,))
def ntk_fn(apply_fn, params, *args):
    # apply_fn needs to be a scalar function
    J = jacobian_fn(apply_fn, params, *args)
    K = jnp.dot(J, J)
    return K


def save_checkpoint(state, path, ckpt_mgr, name=None):
    #workdir = str(workdir+"/ckpt")
    # Create the workdir if it doesn't exist.
    if not os.path.isdir(path):
        os.makedirs(path)

    # Save the checkpoint.
    if jax.process_index() == 0:
        # Get the first replica's state and save it.
        state = jax.device_get(tree_map(lambda x: x[0], state))
        step = int(state.step)
        save_args = flax.training.orbax_utils.save_args_from_target(state)
        ckpt_mgr.save(step, state, save_kwargs={'save_args': save_args})


def restore_checkpoint(state, workdir, step=None):
    # check if passed state is in a sharded state
    # if so, reduce to a single device sharding
    if isinstance(
        jax.tree_map(lambda x: x.sharding, jax.tree_leaves(state.params))[0],
        jax.sharding.PmapSharding,
    ):
        state = jax.tree_map(lambda x: x[0], state)

    # ensuring that we're in a single device setting
    assert isinstance(
        jax.tree_map(lambda x: x.sharding, jax.tree_leaves(state.params))[0],
        jax.sharding.SingleDeviceSharding,
    )
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(workdir, orbax_checkpointer, options)
    #state = checkpoints.restore_checkpoint(workdir, state)
    step = checkpoint_manager.latest_step()  # step = 4
    state = checkpoint_manager.restore(step)
    return state
