#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: akshayjacobthomas
"""

from functools import partial
import jax.numpy as jnp
from jax import jit, grad, vmap

from NN_surrogate.models import SURROGATE
from NN_surrogate.evaluator import BaseEvaluator

from matplotlib import pyplot as plt


class MICRO_SURROGATE(SURROGATE):
    def __init__(self, config):
        super().__init__(config)
        self.n_inputs = config.input_dim
        self.n_outputs = config.output_dim  # Fixed typo: n_ouputs -> n_outputs
        
    def u_net(self, params, x):
        u = self.state.apply_fn(params, x)
        return u

    def residual(self, params, x, y_actual):
        y_pred = self.u_net(params, x)
        return jnp.linalg.norm(y_pred - y_actual)
    
    def losses(self, params, batch_inputs, batch_targets, *args):
        x = batch_inputs
        y_actual = batch_targets
        
        # Fixed parenthesis placement and added explicit in_axes
        batched_residuals = vmap(self.residual, in_axes=(None, 0, 0))(params, x, y_actual)
        mse_loss = jnp.mean(batched_residuals)
        
        #print(mse_loss)
        return {"mse": mse_loss}

class MICRO_SURROGATE_Eval(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

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
