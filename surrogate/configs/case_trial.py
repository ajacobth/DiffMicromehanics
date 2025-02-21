import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "DIFFMICRO"
    wandb.name = "case_trial"
    wandb.tag = None
    
    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "ModifiedMlp"
    arch.num_layers = 4
    arch.hidden_dim = 128
    arch.out_dim = 1
    arch.activation = "tanh"
    #arch.periodicity = False#ml_collections.ConfigDict(
        #{"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    #)
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 128})
    #arch.reparam = ml_collections.ConfigDict(
    #   {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    #)
    
    
    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_epochs =  10
    training.batch_size = 4096

    
    
    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 5e-4
    optim.decay_rate = 0.9
    optim.decay_steps = 5000
    optim.grad_accum_steps = 0


    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({ "mse":1.})
    weighting.momentum = 0.9
    weighting.update_every_steps = 100000000


    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = False # have to make validation dataset
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = True
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 101

    return config
