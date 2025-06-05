from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

from flax.training import train_state
from flax import jax_utils
from flax import linen as nn
from jax.nn.initializers import glorot_normal, normal, zeros, constant, lecun_normal

import jax.numpy as jnp
from jax import lax, jit, grad, pmap, random, tree_map, jacfwd, jacrev
from jax.tree_util import tree_map, tree_reduce, tree_leaves

import optax

from NN_surrogate import archs
from NN_surrogate.utils import flatten_pytree

activation_fn = {
    "relu": nn.relu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "sigmoid": nn.sigmoid,
    "tanh": jnp.tanh,
    "sin": jnp.sin,
}


def _get_activation(str):
    if str in activation_fn:
        return activation_fn[str]

    else:
        raise NotImplementedError(f"Activation {str} not supported yet!")


class TrainState(train_state.TrainState):
    """ Copied from PINN code. Dont need for regularizaztion.
    Will keep constant"""
    weights: Dict
    momentum: float

    def apply_weights(self, weights, **kwargs):
        """Updates `weights` using running average  in return value.

        Returns:
          An updated instance of `self` with new weights updated by applying `running_average`,
          and additional attributes replaced as specified by `kwargs`.
        """

        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree_map(running_average, self.weights, weights)
        weights = lax.stop_gradient(weights)

        return self.replace(
            step=self.step,
            params=self.params,
            opt_state=self.opt_state,
            weights=weights,
            **kwargs,
        )


def _create_arch(config):
    if config.arch_name == "Mlp":
        arch = archs.Mlp(**config)

    elif config.arch_name == "BatchNorm_Mlp":
        arch = archs.BatchNorm_Mlp(**config)


    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def _create_optimizer(config):
    if config.optimizer == "Adam":
        lr = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.decay_steps,
            decay_rate=config.decay_rate,
        )
        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")

    # Gradient accumulation
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)

    return tx


def _create_train_state(config):
    # Initialize network
    arch = _create_arch(config.arch)
    x = jnp.ones(config.input_dim)
    params = arch.init(random.PRNGKey(config.seed), x)

    # Initialize optax optimizer
    tx = _create_optimizer(config.optim)

    # Convert config dict to dict
    init_weights = dict(config.weighting.init_weights)

    state = TrainState.create(
        apply_fn=arch.apply,
        params=params,
        tx=tx,
        weights=init_weights,
        momentum=config.weighting.momentum,
    )

    return jax_utils.replicate(state)


class SURROGATE:
    def __init__(self, config):
        self.config = config
        self.state = _create_train_state(config)
    
    # network predictions
    def u_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")
    
    # strong form loss
    def strong_res_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    # all lossess
    def losses(self, params, batch_inputs, batch_targets, *args):
        raise NotImplementedError("Subclasses should implement this!")


    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch_inputs, batch_targets, *args):
        # Compute losses
        losses = self.losses(params, batch_inputs, batch_targets, *args)
        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss

    @partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch_inputs, batch_targets, *args):
        if self.config.weighting.scheme == "grad_norm":
            # Compute the gradient of each loss w.r.t. the parameters
            grads = jacrev(self.losses)(params, batch_inputs, batch_targets, *args)

            # Compute the grad norm of each loss
            grad_norm_dict = {}
            for key, value in grads.items():
                flattened_grad = flatten_pytree(value)
                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

            # Compute the mean of grad norms over all losses
            mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
            # Grad Norm Weighting
            w = tree_map(lambda x: (mean_grad_norm / x), grad_norm_dict)

        elif self.config.weighting.scheme == "ntk":
            raise NotImplementedError('Not implemented for Additive solver')


        return w
    

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch_inputs, batch_targets,*args):
        
        weights = self.compute_weights(state.params, batch_inputs, batch_targets,
                                       *args)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
        return state
    
    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step(self, state, batch_inputs, batch_targets, *args):
        
        grads = grad(self.loss)(state.params, state.weights,batch_inputs, batch_targets,*args)
        
        grads = lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        return state


class BayesianDense(nn.Module):
    """Bayesian dense layer with mean-field variational weights."""
    features: int
    prior_std: float = 1.0

    @nn.compact
    def __call__(self, x, rng):
        in_dim = x.shape[-1]
        k_mean = self.param("kernel_mean", normal(0.1), (in_dim, self.features))
        k_log_std = self.param("kernel_log_std", constant(-3.0), (in_dim, self.features))
        b_mean = self.param("bias_mean", normal(0.1), (self.features,))
        b_log_std = self.param("bias_log_std", constant(-3.0), (self.features,))

        k_eps = random.normal(rng, k_mean.shape)
        b_eps = random.normal(rng, b_mean.shape)

        kernel = k_mean + jnp.exp(k_log_std) * k_eps
        bias = b_mean + jnp.exp(b_log_std) * b_eps

        y = jnp.dot(x, kernel) + bias

        kl = 0.5 * jnp.sum(
            (jnp.exp(2 * k_log_std) + k_mean ** 2) / (self.prior_std ** 2)
            - 1.0
            + 2.0 * (jnp.log(self.prior_std) - k_log_std)
        )
        kl += 0.5 * jnp.sum(
            (jnp.exp(2 * b_log_std) + b_mean ** 2) / (self.prior_std ** 2)
            - 1.0
            + 2.0 * (jnp.log(self.prior_std) - b_log_std)
        )

        return y, kl


class BayesianMlp(nn.Module):
    """Simple Bayesian MLP using variational dense layers."""
    arch_name: Optional[str] = "BayesianMlp"
    hidden_dim: Sequence[int] = (256, 256, 256, 256)
    out_dim: int = 1
    activation: str = "tanh"
    prior_std: float = 1.0

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x, rng):
        kl_sum = 0.0
        for dim in self.hidden_dim:
            rng, layer_rng = random.split(rng)
            x, kl = BayesianDense(features=dim, prior_std=self.prior_std)(x, layer_rng)
            kl_sum += kl
            x = self.activation_fn(x)

        rng, layer_rng = random.split(rng)
        x, kl = BayesianDense(features=self.out_dim, prior_std=self.prior_std)(x, layer_rng)
        kl_sum += kl
        return x, kl_sum