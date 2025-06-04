import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "train"

    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "MICRO_SURR_BNN"
    wandb.name = "bnn_case"
    wandb.tag = None

    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "BayesianMlp"
    arch.hidden_dim = (128, 128, 128)
    arch.out_dim = 9
    arch.activation = "relu"
    arch.prior_std = 1.0

    config.training = training = ml_collections.ConfigDict()
    training.max_epochs = 5000
    training.batch_size = 4096

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 3000
    optim.grad_accum_steps = 0

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"nll": 1., "kl": 1.})
    weighting.momentum = 0.9
    weighting.update_every_steps = 10000000000

    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 2000
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    config.saving = saving = ml_collections.ConfigDict()
    saving.save_epoch = 99
    saving.num_keep_ckpts = 5

    config.input_dim = 16
    config.output_dim = arch.out_dim
    config.use_train_test_split = True
    config.use_train_val_test_split = True
    config.use_l2reg = False
    config.noise_std = 1.0

    config.seed = 101

    return config
