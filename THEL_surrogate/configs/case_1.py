# configs/case_4.py  – same hyper‑params, now using variable name `config`
# -----------------------------------------------------------------------------
# Training configuration plus inverse‑design block with intuitive feature names.
# -----------------------------------------------------------------------------

import ml_collections
import jax.numpy as jnp  # kept for potential future use


# Feature names – order must match training data csv or json file

# e1	e2	g12	f_nu12	f_nu23	ar	fiber_massfrac	fiber_density	matrix_modulus	matrix_poisson	matrix_density	a11	a22	a12	a13	a23
INPUT_FIELD_NAMES = [
    "e1", "e2", "g12", "f_nu12", "f_cte1",	"f_cte2",
    "f_nu23", "ar", "fiber_massfrac", "fiber_density", "matrix_modulus",
    "matrix_poisson", "matrix_density", "m_cte",
    "a11", "a22", "a12", "a13", "a23", # a33 = 1 - a11 - a22
]

# The size of the inputs is automatically foudn during the code execution

#E1	E2	E3	G12	G13	G23	nu12	nu13	nu23

# k11	k12	k13	k22	k23	k33
OUTPUT_FIELD_NAMES = [
    "E1", "E2", "E3", "G12", "G13", "G23", "nu12","nu13", "nu23",
    "CTE11",	"CTE22", "CTE33",	"CTE12",	"CTE13", "CTE23"
    ]


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
    wandb.project = "THERMOELASTIC_Surrogate"
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
    config.use_train_test_split        = False # I use separate LHS, Its Sobobl now
    config.use_train_val_test_split    = False  # I use separate LHS, Its Sobobl now
    config.use_l2reg = True
    config.seed      = 101


    # ============================================================
    # ----------------- W A R N I N G -------------
    # ============================================================
    
    
    # ============================================================
    # ----------------- W A R N I N G -------------
    # ============================================================
    
    
# ------------NOT CONFIGURED YET -----------------
    # ============================================================
    # Inverse-design block – uses feature names, not indices
    # ============================================================
    config.inverse = inv = ml_collections.ConfigDict()
    
    # ---- ALL 16 inputs get a value ----
    inv.fixed_inputs = {
        # fibre
        "fiber_e1":          240e3,
        "fiber_e2":           14e3,
        "fiber_g12":           28e3,
        "fiber_nu12":          0.2,
        "fiber_nu23":          0.25,
        "fiber_aspect":        10.0,
        "fiber_massfrac":       0.25,
        "fiber_density":     1800.0,
        # matrix
        "matrix_modulus":      2.34e3,
        "matrix_poissonratio": 0.35,
        "matrix_density":    1350.0,
        # coupling / angle terms  (free but still given as a *starting guess*)
        "a11": 0.7,
        "a22": 0.2,
        "a12": 0.0,
        "a13": 0.0,
        "a23": 0.0,
    }
    
    # Names of inputs to optimise
    inv.free_inputs = [
                       "matrix_modulus",
                       "matrix_poissonratio",
                       "a11",
                       "a12",
                       "fiber_aspect",
                       "a22"]
    # Bounds for each free variable (lo, hi)
    
    inv.bounds = {"fiber_e1":(180e3, 300e3),
        "matrix_modulus": (1.0e3, 5.0e3),
        "matrix_poissonratio": (0.25, 0.4),
        "fiber_aspect": (5., 30.),
        "a11":            (0.3,    1.0),
        "a22":            (0.0,    0.6),
        "a12":(-0.1, 0.1)
    }
    # Desired surrogate outputs
    inv.target_outputs = {
        "E1": 16.92e3,  # MPA
        "E2": 4.85e3,
        "nu13":  0.35,
    }
    
    # Solver hyper-params
    inv.optim          = "adam" # jaxopt for lbfgs  # or "adam" optax for optimization
    inv.lbfgs_maxiter  = 200
    inv.lbfgs_tol      = 1e-6
    
    
    # inv.init_free    = [0.0, 0.0, 0.0, 0.0]
    # inv.adam_lr      = 1e-2
    # inv.adam_steps   = 500

    # Uncomment if you’ll only run inverse‑design
    # config.mode = "inverse"

    return config
