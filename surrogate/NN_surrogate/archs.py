from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Union, Dict
from flax import linen as nn
import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from flax.linen.dtypes import promote_dtype
from jax.nn.initializers import glorot_normal, normal, zeros, constant

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
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
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
        for dim in self.hidden_dims:
            x = Dense(features=dim)(x)
            x = self.activation_fn(x)

        x = Dense(features=self.out_dim)(x)
        return x

