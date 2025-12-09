#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: akshayjacobthomas
"""

from functools import partial
import jax.numpy as jnp
import jax
from jax import jit, grad, vmap, tree_map

from NN_surrogate.models import SURROGATE
from NN_surrogate.evaluator import BaseEvaluator

from matplotlib import pyplot as plt


class MICRO_SURROGATE(SURROGATE):
    def __init__(self, config, input_mean=None, input_std=None, 
                 output_mean=None, output_std=None, 
                 x_val=None, y_val=None):
        super().__init__(config)
        self.n_inputs = config.input_dim
        self.n_outputs = config.output_dim  # Fixed typo: n_ouputs -> n_outputs
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.x_val = x_val
        self.y_val = y_val
        
        
    def u_net(self, params, x):
        u = self.state.apply_fn(params, x)
        return u

    def residual(self, params, x, y_actual):
        y_pred = self.u_net(params, x)
        return (y_pred - y_actual)**2
    
    def losses(self, params, batch_inputs, batch_targets, *args):
        x = batch_inputs
        y_actual = batch_targets
    
        # Compute squared error per output dimension
        batched_residuals = vmap(self.residual, in_axes=(None, 0, 0))(params, x, y_actual)
        
        # Take mean over batch and output dimensions separately
        mse_loss_per_output = jnp.mean(batched_residuals, axis=0)  # Mean across batch
        mse_loss = jnp.mean(mse_loss_per_output)  # Final mean across outputs
        
        return {"mse": mse_loss}
    
    
    @partial(jit, static_argnums=(0,))
    def compute_validation_error(self, params):
        # Normalize test inputs using stored stats
        test_inputs = (self.x_val - self.input_mean) / self.input_std
    
        # Model predictions
        test_preds = self.u_net(params, test_inputs)
    
        # Denormalize predictions to original scale
        y_scaled = (self.y_val - self.output_mean) / self.output_std
        #test_preds = (test_preds * self.output_std) + self.output_mean
        
        mse = jnp.mean((test_preds - y_scaled) ** 2)  # Mean Squared Error
        
        return mse
    
    
class MICRO_SURROGATE_L2(SURROGATE):
    def __init__(self, config, input_mean=None, input_std=None, 
                 output_mean=None, output_std=None, 
                 x_val=None, y_val=None):
        super().__init__(config)
        self.n_inputs = config.input_dim
        self.n_outputs = config.output_dim  # Fixed typo: n_ouputs -> n_outputs
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.x_val = x_val
        self.y_val = y_val
        
        
    def u_net(self, params, x):
        u = self.state.apply_fn(params, x)
        return u

    def residual(self, params, x, y_actual):
        y_pred = self.u_net(params, x)
        return (y_pred - y_actual)**2
    
    def l2_regularization(self, params):
        """Compute L2 regularization term for model parameters."""
        #return jnp.sum(jnp.array([
        #    jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params) if p.dtype == jnp.float32
        #]))
        # This will be safely usable inside a jitted loss
        def leaf_l2(p):
            # Only act on floating-point arrays
            if not jnp.issubdtype(p.dtype, jnp.floating):
                return 0.0
        
            # Treat 1D arrays as biases (skip them)
            if p.ndim == 1:
                return 0.0
        
            return jnp.sum(p**2)
        
        # Map over the whole param tree
        sq_sums_tree = jax.tree_util.tree_map(leaf_l2, params)
        # Collect and sum
        return jnp.sum(jnp.array(jax.tree_util.tree_leaves(sq_sums_tree)))


    
    def losses(self, params, batch_inputs, batch_targets, *args):
        x = batch_inputs
        y_actual = batch_targets
    
        # Compute squared error per output dimension
        batched_residuals = vmap(self.residual, in_axes=(None, 0, 0))(params, x, y_actual)
        
        # Take mean over batch and output dimensions separately
        mse_loss_per_output = jnp.mean(batched_residuals, axis=0)  # Mean across batch
        mse_loss = jnp.mean(mse_loss_per_output)  # Final mean across outputs
    
        
    
        l2_loss = self.l2_regularization(params)
    
        return {"mse": mse_loss, "l2":l2_loss}
    
    
    @partial(jit, static_argnums=(0,))
    def compute_validation_error(self, params):
        # Normalize test inputs using stored stats
        test_inputs = (self.x_val - self.input_mean) / self.input_std
    
        # Model predictions
        test_preds = self.u_net(params, test_inputs)
    
        # Denormalize predictions to original scale
        y_scaled = (self.y_val - self.output_mean) / self.output_std
        #test_preds = (test_preds * self.output_std) + self.output_mean
        
        mse = jnp.mean((test_preds - y_scaled) ** 2)  # Mean Squared Error

        return mse


class MICRO_SURROGATE_BNN(SURROGATE):
    """Bayesian neural network surrogate using variational inference."""

    def __init__(self, config, dataset_size, input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 x_val=None, y_val=None):
        super().__init__(config)
        self.n_inputs = config.input_dim
        self.n_outputs = config.output_dim
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.x_val = x_val
        self.y_val = y_val
        self.dataset_size = dataset_size
        self.noise_std = getattr(config, "noise_std", 1.0)

    def u_net(self, params, x, rng):
        y, kl = self.state.apply_fn(params, x, rng)
        return y, kl

    def losses(self, params, batch_inputs, batch_targets, rng):
        preds, kl = self.u_net(params, batch_inputs, rng)
        mse = jnp.mean((preds - batch_targets) ** 2)
        nll = mse / (2 * self.noise_std ** 2)
        kl = kl / self.dataset_size
        return {"nll": nll, "kl": kl}

    @partial(jit, static_argnums=(0,))
    def compute_validation_error(self, params):
        test_inputs = (self.x_val - self.input_mean) / self.input_std
        rng = jax.random.PRNGKey(0)
        preds, _ = self.u_net(params, test_inputs, rng)
        y_scaled = (self.y_val - self.output_mean) / self.output_std
        mse = jnp.mean((preds - y_scaled) ** 2)
        return mse
    
class MICRO_SURROGATE_BNN(SURROGATE):
    """Bayesian neural network surrogate using variational inference."""

    def __init__(self, config, dataset_size, input_mean=None, input_std=None,
                 output_mean=None, output_std=None,
                 x_val=None, y_val=None):
        super().__init__(config)
        self.n_inputs = config.input_dim
        self.n_outputs = config.output_dim
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
        self.x_val = x_val
        self.y_val = y_val
        self.dataset_size = dataset_size
        self.noise_std = getattr(config, "noise_std", 1.0)

    def u_net(self, params, x, rng):
        y, kl = self.state.apply_fn(params, x, rng)
        return y, kl

    def losses(self, params, batch_inputs, batch_targets, rng):
        preds, kl = self.u_net(params, batch_inputs, rng)
        mse = jnp.mean((preds - batch_targets) ** 2)
        nll = mse / (2 * self.noise_std ** 2)
        kl = kl / self.dataset_size
        return {"nll": nll, "kl": kl}

    @partial(jit, static_argnums=(0,))
    def compute_validation_error(self, params):
        test_inputs = (self.x_val - self.input_mean) / self.input_std
        rng = jax.random.PRNGKey(0)
        preds, _ = self.u_net(params, test_inputs, rng)
        y_scaled = (self.y_val - self.output_mean) / self.output_std
        mse = jnp.mean((preds - y_scaled) ** 2)
        return mse


class MICRO_SURROGATE_Eval(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        val_error = self.model.compute_validation_error(params)
        self.log_dict["val_error"] = val_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.t_star, self.model.x_star)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch_inputs, batch_targets, *args, u_ref=None):
        self.log_dict = super().__call__(state,  batch_inputs, batch_targets, *args)
        
        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
