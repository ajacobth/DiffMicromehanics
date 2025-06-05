from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict
from flax import linen as nn
import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from jax.nn.initializers import glorot_normal, normal, zeros, constant, lecun_normal

Dtype = Any # this could be any for now, lets just keep it for now

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



class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", lecun_normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    dtype: Optional[Dtype] = None

    @nn.compact
    def __call__(self, x):
        
        kernel = self.param(
            "kernel", self.kernel_init, (x.shape[-1], self.features))
        
        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y



class Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    hidden_dim: Sequence[int] = (256, 256, 256, 256)  # Default 4 layers
    out_dim: int = 1
    activation: str = "tanh"
    fourier_emb: Union[None, Dict] = None

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        # Loop through each specified hidden dimension
        for dim in self.hidden_dim:
            x = Dense(features=dim)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim)(x)
        return x



class BatchNorm_Mlp(nn.Module):
    arch_name: Optional[str] = "Mlp"
    hidden_dim: Sequence[int] = (256, 256, 256, 256)  # Default 4 layers
    out_dim: int = 1
    activation: str = "tanh"
    fourier_emb: Union[None, Dict] = None
    use_batchnorm: bool = True  # Add a flag for BatchNorm usage

    def setup(self):
        self.activation_fn = _get_activation(self.activation)

    @nn.compact
    def __call__(self, x):
        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        # Loop through each hidden layer
        for dim in self.hidden_dim:
            x = nn.Dense(features=dim)(x)
            x = nn.BatchNorm(x)
            x = self.activation_fn(x)

        x = nn.Dense(features=self.out_dim)(x)
        return x


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
