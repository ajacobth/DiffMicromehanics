# configs/case_4.py  – same hyper‑params, now using variable name `config`
# -----------------------------------------------------------------------------
# Training configuration plus inverse‑design block with intuitive feature names.
# -----------------------------------------------------------------------------

import ml_collections
import jax.numpy as jnp  # kept for potential future use


# Feature names – order must match training data csv or json file

INPUT_FIELD_NAMES = [
    "k_f1", "k_f2", "k_m", "ar_f",
    "w_f", "rho_f", "rho_m",
    "a11", "a22", "a12", "a13", "a23", # a33 = 1 - a11 - a22
]

# The size of the inputs is automatically foudn during the code execution

#k_f1	k_f2	k_m	ar_f	w_f	rho_f	rho_m	a11	a22	a12	a13	a23

# k11	k12	k13	k22	k23	k33
OUTPUT_FIELD_NAMES = [
    "k11", "k12", "k13", "k22", "k23",
    "k33"]


def get_config():
    # ----------------------------------------------------------------
    # Create root config dict
    # ----------------------------------------------------------------
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------
    # Training‑time settings (unchanged)
    # ------------------------------------------------------------
    config.mode = "train"

    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "TC_Surrogate"
    wandb.name    = "case_1"
    wandb.tag     = None

    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name  = "Mlp"
    arch.hidden_dim = (128, 512, 256, 256, 128, 64, 16)
    arch.out_dim    = len(OUTPUT_FIELD_NAMES)      # 6
    arch.activation = "relu"

    config.training = training = ml_collections.ConfigDict()
    training.max_epochs = 501
    training.batch_size = 512

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer       = "Adam"
    optim.beta1, optim.beta2 = 0.9, 0.999
    optim.eps             = 1e-8
    optim.learning_rate   = 1e-3
    optim.decay_rate      = 0.9
    optim.decay_steps     = 1e6
    optim.grad_accum_steps = 0

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm" # None
    weighting.init_weights = ml_collections.ConfigDict({"mse": 1., "l2": 1e-6})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1e12 # This way weights are not udpated

    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 1 # 1
    logging.print_every_epochs = 1
    logging.log_errors      = True
    logging.log_losses      = True
    logging.log_weights     = True
    logging.log_preds       = False
    logging.log_grads       = False
    logging.log_ntk         = False

    config.saving = saving = ml_collections.ConfigDict()
    saving.save_epoch      = 25
    saving.num_keep_ckpts  = 5

    # Misc
    config.input_dim  = len(INPUT_FIELD_NAMES)   # 12
    config.output_dim = len(OUTPUT_FIELD_NAMES)  # 6
    config.use_train_test_split        = False # I use separate LHS
    config.use_train_val_test_split    = False  # I use separate LHS
    config.use_l2reg = True
    config.seed      = 101


    # ============================================================
    # Inverse-design block – uses feature names, not indices
    # Inputs:  k_f1, k_f2, k_m, ar_f, w_f, rho_f, rho_m, a11, a22, a12, a13, a23
    # Outputs: k11, k12, k13, k22, k23, k33
    # ============================================================
    config.inverse = inv = ml_collections.ConfigDict()

    # ---- All 12 inputs must appear in either fixed_inputs or free_inputs ----
    inv.fixed_inputs = {
        # Densities (usually known from material datasheet)
        "rho_f":  1800.0,   # kg/m^3 – fibre density
        "rho_m":  1200.0,   # kg/m^3 – matrix density
        # Orientation tensors (fixed; set by processing conditions)
        "a11":  0.7,
        "a22":  0.2,
        "a12":  0.0,
        "a13":  0.0,
        "a23":  0.0,
        # Starting guesses for free variables (also needed here so assemble_x works)
        "k_f1": 10.0,
        "k_f2":  5.0,
        "k_m":   0.2,
        "ar_f": 20.0,
        "w_f":   0.30,
    }

    # Names of constituent properties to recover
    inv.free_inputs = [
        "k_f1",   # fibre thermal conductivity (axial)
        "k_f2",   # fibre thermal conductivity (transverse)
        "k_m",    # matrix thermal conductivity
        "ar_f",   # fibre aspect ratio
        "w_f",    # fibre weight fraction
    ]

    # Bounds for each free variable (lo, hi)
    inv.bounds = {
        "k_f1": (1.0,  200.0),   # W/(m·K)
        "k_f2": (0.5,  100.0),
        "k_m":  (0.01,   5.0),
        "ar_f": (5.0,  500.0),
        "w_f":  (0.01,   0.65),
    }

    # Target composite thermal conductivities (W/(m·K))
    # These are the NN outputs you want to match
    inv.target_outputs = {
        "k11": 1.35,
        "k22": 2.21,
        "k33": 0.58,
    }

    # Solver hyper-params
    inv.optim         = "lbfgsb"  # lbfgs / lbfgsb / adam
    inv.lbfgs_maxiter = 300
    inv.lbfgs_tol     = 1e-9

    # inv.adam_lr    = 1e-2
    # inv.adam_steps = 2000
    # inv.init_free  = [10.0, 5.0, 0.2, 20.0, 0.30]  # override starting guess

    return config
