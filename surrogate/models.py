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
        self.n_inputs= config.input_dim
        self.n_ouputs = config.output_dim
        
    def u_net(self, params, x):
        u = self.state.apply_fn(params, x)
        return u

    
    def residual(self, params, x, y_actual):
        y_pred = self.u_net(params, x)
        
        return jnp.linalg.norm(y_pred-y_actual)
    
                 
    def losses(self, params, batch, *args):
        # args[0] is step number - required for sequential collocation sampling
        # Initial condition loss
        x = batch[:, :self.n_inputs]
        y_actual = batch[:, self.n_inputs:]
        mse_loss = jnp.mean(vmap(self.residual, (None, 0, 0)))(params, x, y_actual)
        loss_dict = {"mse": mse_loss}
        return loss_dict


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

    def __call__(self, state, time_batch, batch_initial, *args, u_ref=None):
        self.log_dict = super().__call__(state,  time_batch, batch_initial, *args)

        if self.config.weighting.use_causal:
            raise NotImplementedError('causal weighting not implemented for A3D')

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
