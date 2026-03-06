"""export_model.py – package a trained elastic surrogate for deployment in final/.

Run this after training to update the artifacts in final/models/elastic/:
    - model_config.json  (arch + field names)
    - normalization_stats.npz
    - ckpt/<checkpoint_name>/

Usage
-----
    python export_model.py --config configs/case_5.py --workdir . --out ../final/models/elastic
"""
from __future__ import annotations

import json
import os
import shutil

from absl import app, flags
from ml_collections import config_flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "workdir", default=".",
    help="Training workdir containing ckpt/ and normalization_stats.npz",
)
flags.DEFINE_string(
    "out", default="../final/models/elastic",
    help="Destination directory (inside final/models/)",
)
config_flags.DEFINE_config_file(
    "config", default="./configs/case_5.py", lock_config=True,
)

INPUT_FIELD_NAMES = [
    "e1", "e2", "g12", "f_nu12",
    "f_nu23", "ar", "fiber_massfrac", "fiber_density",
    "matrix_modulus", "matrix_poisson", "matrix_density",
    "a11", "a22", "a12", "a13", "a23",
]
OUTPUT_FIELD_NAMES = [
    "E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23",
]


def main(_argv):
    cfg     = FLAGS.config
    workdir = os.path.abspath(FLAGS.workdir)
    out_dir = os.path.abspath(FLAGS.out)
    os.makedirs(out_dir, exist_ok=True)

    # 1. model_config.json
    model_config = {
        "_note": f"Generated from {cfg.wandb.name}. Update checkpoint_name if you retrain.",
        "arch_name":       cfg.arch.arch_name,
        "hidden_dim":      list(cfg.arch.hidden_dim),
        "activation":      cfg.arch.activation,
        "use_l2reg":       bool(getattr(cfg, "use_l2reg", True)),
        "checkpoint_name": cfg.wandb.name,
        "input_fields":    INPUT_FIELD_NAMES,
        "output_fields":   OUTPUT_FIELD_NAMES,
    }
    config_path = os.path.join(out_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Wrote  {config_path}")

    # 2. normalization_stats.npz
    src_stats = os.path.join(workdir, "normalization_stats.npz")
    dst_stats = os.path.join(out_dir, "normalization_stats.npz")
    shutil.copy2(src_stats, dst_stats)
    print(f"Copied {dst_stats}")

    # 3. checkpoint directory
    ckpt_name = cfg.wandb.name
    src_ckpt  = os.path.join(workdir, "ckpt", ckpt_name)
    dst_ckpt  = os.path.join(out_dir,  "ckpt", ckpt_name)
    if not os.path.isdir(src_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {src_ckpt}")
    if os.path.exists(dst_ckpt):
        shutil.rmtree(dst_ckpt)
    shutil.copytree(src_ckpt, dst_ckpt)
    print(f"Copied {dst_ckpt}")

    print(f"\nDone. Elastic surrogate artifacts written to:\n  {out_dir}")


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
