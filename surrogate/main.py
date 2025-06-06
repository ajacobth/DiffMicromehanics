import os
# Deterministic
#os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # Ensures deterministic behavior
#os.environ["JAX_PLATFORM_NAME"] = "METAL"  # Must be uppercase!
from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

import jax

import train
import eval
import gui_predict  # Import GUI script

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

def main(argv):
    if FLAGS.config.mode == "train":
        train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "gui":
        gui_predict.gui(FLAGS.config, FLAGS.workdir)  # Run GUI when mode is "gui"

if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
