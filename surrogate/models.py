#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:18:52 2024

@author: akshayjacobthomas
"""

from functools import partial
import jax.numpy as jnp
from jax import jit, grad, vmap

from A3DPINN.models import ForwardIVP
from A3DPINN.evaluator import BaseEvaluator
#from A3DPINN.utils import ntk_fn, flatten_pytree
from A3DPINN.samplers import SeqCollocationSampler, SeqNeumanCollocationSampler_B1, SeqNeumanCollocationSampler_B2
from A3DPINN.samplers import NeumannInitialSampler, SeqInitialBoundarySampler

from matplotlib import pyplot as plt


class A3DHeatTransfer(ForwardIVP):
    def __init__(self, config):
        super().__init__(config)
        
        self.u_max = config.process_conditions.deposition_temperature
        self.t_max = config.dimensions.t_max
        self.x_max = config.dimensions.x_max
        self.y_max = config.dimensions.y_max
        
        # processing conditions in scaled version
        self.deposition_temperature_scaled = config.process_conditions.deposition_temperature/self.u_max
        self.bed_temperature_scaled = config.process_conditions.bed_temperature/self.u_max
        
        # define test space
        #currenlty same definitions in x, y, and t
        #% -_-_-_ NOT BEING USED IN THIS CASE _-_-_-_%#
        
        
        #define material properties
        self.rho = config.material_properties.density
        self.K_xx = config.material_properties.thermal_conductivity_xx
        self.K_yy = config.material_properties.thermal_conductivity_yy
        self.C = config.material_properties.specific_heat
        self.h = config.material_properties.heat_transfer_coefficient
        
        self.alpha_xx = self.K_xx/(self.rho*self.C)
        self.alpha_yy = self.K_yy/(self.rho*self.C)
        
        
        # introduce scaled versions of material propeerties 
        self.K_xx_scaled = self.K_xx/self.x_max
        self.K_yy_scaled = self.K_yy/self.y_max
        
        self.alpha_xx_scaled = self.alpha_xx * self.t_max/(self.x_max**2)
        self.alpha_yy_scaled = self.alpha_yy * self.t_max/(self.y_max**2)
    
            
        #define sacled versions of procesisng contions
        self.print_speed_speed_scaled = config.process_conditions.print_speed *self.t_max/(self.x_max)
        self.init_length_scaled = config.process_conditions.init_length/self.x_max
        self.bead_width_scaled = config.process_conditions.bead_width/self.y_max
        self.velocity_vector = config.process_conditions.velocity_vector
        
        self.ambient_convection_temp = config.process_conditions.ambient_convection_temp/self.u_max
        self.ambient_radiation_temp = config.process_conditions.ambient_radiation_temp/self.u_max
        
        #sampler - sequential collocation sampler
        self.seqSampler = SeqCollocationSampler(config.training.batch_size_per_device, 
                                                self.init_length_scaled, 
                                                self.print_speed_speed_scaled*self.velocity_vector,
                                                self.bead_width_scaled)
        
        self.seqSamplerNeumann_B1 = SeqNeumanCollocationSampler_B1(config.training.batch_size_per_device, 
                                                self.init_length_scaled, 
                                                self.print_speed_speed_scaled*self.velocity_vector,
                                                self.bead_width_scaled)
        
        self.seqSamplerNeumann_B2 = SeqNeumanCollocationSampler_B2(config.training.batch_size_per_device, 
                                                self.init_length_scaled, 
                                                self.print_speed_speed_scaled*self.velocity_vector,
                                                self.bead_width_scaled)
        
        self.seqSamplerNeumannInitial= NeumannInitialSampler(config.training.batch_size_per_device, 
                                                self.init_length_scaled, 
                                                self.print_speed_speed_scaled*self.velocity_vector,
                                                self.bead_width_scaled)
        
        self.seqInitialBoundarySampler = SeqInitialBoundarySampler(config.training.batch_size_per_device, 
                                                self.init_length_scaled, 
                                                self.print_speed_speed_scaled*self.velocity_vector,
                                                self.bead_width_scaled)
        # Predictions over a grid
        self.u_pred_fn = vmap(self.u_net, (None, None, 0, 0))
        #self.r_pred_fn = vmap(vmap(self.r_net_weak, (None, None, 0)), (None, 0, None))
        self.delta_time_lossMap= vmap(self.strong_res_net, (None, 0, 0, 0))
        self.delta_time_NeumannLossMap_b1 = vmap(self.loss_neumann_b1, (None, 0, 0, 0))
        self.delta_time_NeumannLossMap_b2 = vmap(self.loss_neumann_b2, (None, 0, 0, 0))
        
        self.delta_timeIntial_NeumannLossMap = vmap(self.loss_neumann_initial, (None, 0, 0, 0))
        self.delta_timeEvolving_InitialLossMap = vmap(self.evolving_initial, (None, 0, 0, 0))
        # for evaluation
        self.evalfn_ = vmap(self.u_net, (None, 0, None, None))

        self.bs = config.training.batch_size_per_device
    def u_net(self, params, t, x, y):
        z = jnp.stack([t,x,y])#jnp.concatenate((jnp.concatenate((t,x), axis=0), y), axis=0)
        u = self.state.apply_fn(params, z)
        return u[0]

    
    def strong_res_net(self, params, t, x, y):
        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y)
        
        return self.alpha_xx_scaled*u_xx +self.alpha_yy_scaled*u_yy - u_t
    
    def loss_neumann_b1(self, params, t, x, y):
        
        # first iteration of evolving neumann BC
        u = self.u_net(params, t, x, y)
        #u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        
        return self.K_yy_scaled*u_y + self.h*u # this is after multiplying wiht unit vector
    
    
    def loss_neumann_b2(self, params, t, x, y):
                
        u_sol = self.bed_temperature_scaled
        u_pred = self.u_net(params, t, x, y)
        return u_pred - u_sol
            
    def loss_neumann_initial(self, params, t, x, y):
        
        # first iteartion of evolving Neumann BC initial
        u_sol = self.deposition_temperature_scaled

        u_pred= self.u_net(params, t, x, y)
        
        
        return u_sol-u_pred
        
    def evolving_initial(self, params, t, x, y):
        u_sol = self.deposition_temperature_scaled
        u_pred= self.u_net(params, t, x, y)
        
        return u_sol-u_pred
    
    @partial(jit, static_argnums=(0,))
    def delta_time_loss(self, params, step, time):
        batch  = self.seqSampler(step[0], time)
        
        #here batch[:, 1] are the x-coordinates and batch[:, 2] are y-coordinates
        res = self.delta_time_lossMap(params,batch[:, 0], batch[:, 1], batch[:, 2])
        return jnp.mean(res**2)
    
    @partial(jit, static_argnums=(0,))
    def delta_time_neumann_loss_b1(self, params, step, time):
        batch = self.seqSamplerNeumann_B1(step[0], time)
        
        res_neurmann = self.delta_time_NeumannLossMap_b1(params,batch[:, 0], batch[:, 1], batch[:, 2])
        #here batch[:, 1] are the x-coordinates and batch[:, 2] are y-coordinates
        
        return jnp.mean(res_neurmann**2)
    
    
    @partial(jit, static_argnums=(0,))
    def delta_time_neumann_loss_b2(self, params, step, time):
        batch = self.seqSamplerNeumann_B2(step[0], time)
        
        res_neurmann = self.delta_time_NeumannLossMap_b2(params,batch[:, 0], batch[:, 1], batch[:, 2])
        #here batch[:, 1] are the x-coordinates and batch[:, 2] are y-coordinates
        
        return jnp.mean(res_neurmann**2)
    
    @partial(jit, static_argnums=(0,))
    def delta_time_neumann_initial_loss(self, params, step, time):
        batch = self.seqSamplerNeumannInitial(step[0], time)
        
        res_neurmann_initial = self.delta_timeIntial_NeumannLossMap(params,batch[:, 0], batch[:, 1], batch[:, 2])
        
        
        return jnp.mean(res_neurmann_initial**2)
    
    @partial(jit, static_argnums=(0,))
    def delta_time_evolving_initial_loss(self, params, step, time):
        batch = self.seqInitialBoundarySampler(step[0], time)
        
        res_evolving_initial = self.delta_timeEvolving_InitialLossMap(params,batch[:, 0], batch[:, 1], batch[:, 2])
        
        return jnp.mean(res_evolving_initial**2)
    
    ###### _ NON EVOLVING LOSSES ########
        
    @partial(jit, static_argnums=(0,))
    def loss_boundary(self, params, t, x, y):
        u_sol = self.bed_temperature_scaled
        u_pred = self.u_net(params, t, x, y)
        return (u_pred - u_sol)**2 
    
    @partial(jit, static_argnums=(0,))
    def loss_initial(self, params, t, x, y): #this is actually neumann/adiabatic
        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        return (self.K_xx_scaled*u_x - 0.*self.u_net(params, t, x, y))**2 
                    
                 
    def losses(self, params, time_batch, batch_initial, *args):
        # args[0] is step number - required for sequential collocation sampling
        # Initial condition loss
        
        ics_loss = jnp.mean(vmap(self.loss_initial, (None, 0, 0, 0))(params, batch_initial[:, 0], batch_initial[:, 1], batch_initial[:, 2]))
        
        #bcs_loss = jnp.mean(vmap(self.loss_boundary, (None, 0, 0, 0))(params, batch_boundary[:, 0], batch_boundary[:, 1], batch_boundary[:, 2]))
        
        
        
        # Residual loss
        if self.config.weighting.use_causal == True:
            return NotImplementedError('Not implemented for Additive')
            #l, w = self.res_and_w(params, batch)
            #res_loss = jnp.mean(l * w)
            
        if self.config.training.loss_type == "strong":
            #time_batch = jnp.concatenate((time_batch1, time_batch2), axis=0)
            #res = vmap(self.strong_res_net, (None, 0, 0, 0))(params, batch_initial[:, 0], batch_initial[:, 1], batch_initial[:, 2])**2
            
            # note taht the avrage here is over time samples - expectation
            res_loss = jnp.mean(vmap(self.delta_time_loss, (None, 0, 0))(params, args[0], time_batch))
            
            dbc_loss_b2 = jnp.mean(vmap(self.delta_time_neumann_loss_b2, (None, 0, 0))(params, args[0], time_batch))
            ncs_loss_b1 = jnp.mean(vmap(self.delta_time_neumann_loss_b1, (None, 0, 0))(params, args[0], time_batch))
            
            #ncs_loss_init = jnp.mean(vmap(self.delta_time_neumann_initial_loss, (None, 0, 0))(params, args[0], time_batch))
            
            evolving_init = jnp.mean(vmap(self.delta_time_evolving_initial_loss, (None, 0, 0))(params, args[0], time_batch))
            
        elif self.config.training.loss_type == "weak":
            return NotImplementedError('Weak form error not implemented for Additive')
            #r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
            #res_loss = jnp.mean((r_pred) ** 2)

        #loss_dict = {"ics":ics_loss, "dbc_b1": dbc_loss_b1,"ncs_b2": ncs_loss_b2,"evol_init":evolving_init ,"res": res_loss}
        loss_dict = {"wall": ics_loss, "dbc_b1": dbc_loss_b2,"nbc_b2": ncs_loss_b1,"evol_init":evolving_init ,"res": res_loss}
        return loss_dict


    #@partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        return NotImplementedError('L2 error not implemented yet')
        
    def evaluate_Uplot(self, params, time):
        
        length_updated = self.velocity_vector[0]*self.print_speed_speed_scaled*time + self.init_length_scaled[0]
        width_updated = self.bead_width_scaled
        #batch = self.seqSampler(1, time)
        x_batch = jnp.linspace(0., length_updated, 200) # batch[:, 1]

        y_batch = jnp.linspace(0., width_updated, 200) # batch[:, 2]#
        
        xx, yy = jnp.meshgrid(x_batch, y_batch)
        x_volume = jnp.concatenate((xx.reshape(-1)[:,None], yy.reshape(-1)[:,None]), axis=1)
        
        temp_scaled = self.u_pred_fn(params, time, x_volume[:, 0], x_volume[:, 1])
        
        return temp_scaled*self.u_max, x_volume[:, 0]*self.x_max, x_volume[:, 1]*self.y_max
    
    
    def evaluate_init_plot(self, params):
        """ Evaluate the temeprature in the entire bead at time = 0 """
        
        x_batch  = jnp.linspace(0., 1., 200)
        y_batch = jnp.linspace(0., 1., 200)
        
        xx, yy = jnp.meshgrid(x_batch, y_batch)
        x_volume = jnp.concatenate((xx.reshape(-1)[:,None], yy.reshape(-1)[:,None]), axis=1)
        
        temp_scaled = self.u_pred_fn(params, 0., x_volume[:, 0], x_volume[:, 1])
        return temp_scaled*self.u_max, x_volume[:, 0]*self.x_max, x_volume[:, 1]*self.y_max
        
        

class A3DHeatTransferEvaluator(BaseEvaluator):
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
