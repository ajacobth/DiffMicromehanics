import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"


    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "MICRO_SURR_data12k"
    wandb.name = "case_1"
    wandb.tag = None
    
    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.hidden_dim = (128, 256, 128, 128, 64, 64, 32, 32)
    arch.out_dim = 9
    arch.activation = "relu"
    
    # data file
    #config.data.path = "data.csv"
    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_epochs =  10000
    training.batch_size = 8192

    
    
    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 3000
    optim.grad_accum_steps = 0


    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({ "mse":1.})
    weighting.momentum = 0.9
    weighting.update_every_steps = 10000000000


    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 2000
    logging.log_errors = True # have to make validation dataset
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_epoch = 99
    saving.num_keep_ckpts = 5

    # Input shape for initializing Flax models and training split and regulari
    config.input_dim = 16
    config.output_dim = arch.out_dim
    config.use_train_test_split = True
    config.use_train_val_test_split = True
    config.use_l2reg = False

    # Integer for PRNG random seed.
    config.seed = 101

    return config
