"""
run_inverse_temp.py
===================
Entry point for gradient-based inverse estimation of constituent thermal
conductivities as functions of temperature.

The trained TC surrogate NN is used as the forward model to evaluate composite
conductivities K11, K22, K33.  All problem parameters are read from the
``inverse_temp`` block of the config file — no other command-line arguments
are needed.

Usage
-----
    python run_inverse_temp.py --config ./configs/case_1.py --workdir .
"""

from __future__ import annotations

import os
import argparse
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend — no display needed
import matplotlib.pyplot as plt

# ── JAX setup must happen before any JAX/model imports ───────────────────────
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
jax.config.update("jax_enable_x64", True)

from inverse_estimation_temp import (
    ConstituentParams,
    PolymerConductivityModel,
    FiberConductivityModel,
    compute_composite_conductivity,
    run_inverse_estimation,
)


# =============================================================================
# Config loader
# =============================================================================

def load_config(config_path: str):
    """
    Load an ml_collections ConfigDict from a Python config file.

    Parameters
    ----------
    config_path : str
        Path to a Python file exposing a ``get_config()`` function.

    Returns
    -------
    config : ml_collections.ConfigDict
    """
    spec = importlib.util.spec_from_file_location("_cfg_module", config_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config()


# =============================================================================
# NN predictor factory
# =============================================================================

def create_nn_predictor(config, workdir: str):
    """
    Load the trained TC surrogate and return a batched forward-pass callable.

    Parameters
    ----------
    config  : ml_collections.ConfigDict
    workdir : str — directory containing ``ckpt/`` and ``normalization_stats.npz``

    Returns
    -------
    fwd : callable
        Accepts an (N, 12) array-like and returns an (N, 6) array-like
        [k11, k12, k13, k22, k23, k33].
    """
    import jax.numpy as jnp
    from NN_surrogate.utils import restore_checkpoint
    import models

    model  = models.MICRO_SURROGATE_L2(config)
    params = restore_checkpoint(
        model.state, os.path.join(workdir, "ckpt", config.wandb.name)
    )["params"]

    stats   = np.load(os.path.join(workdir, "normalization_stats.npz"))
    mu_in   = jnp.array(stats["input_mean"])
    sig_in  = jnp.array(stats["input_std"])
    mu_out  = jnp.array(stats["target_mean"])
    sig_out = jnp.array(stats["target_std"])

    @jax.jit
    def fwd(x):
        """Normalise inputs → NN → denormalise outputs."""
        xb  = jnp.array(x)
        y_s = model.u_net(params, (xb - mu_in) / sig_in)
        return y_s * sig_out + mu_out  # (N, 6)

    return fwd


# =============================================================================
# Data loading
# =============================================================================

def load_data(data_path: str):
    """
    Load temperature and composite conductivity measurements from CSV or Excel.

    Column names are matched case-insensitively.  Missing K columns are
    silently skipped — the optimiser uses only those present.

    Parameters
    ----------
    data_path : str
        Path to a ``.csv`` or ``.xlsx`` file with columns
        [Temperature, K11, K22, K33].

    Returns
    -------
    temperatures : np.ndarray, shape (N,)
    K_data       : dict  keys 'K11','K22','K33' → np.ndarray (N,) or None
    """
    ext = os.path.splitext(data_path)[1].lower()
    df  = pd.read_excel(data_path) if ext in (".xlsx", ".xls") else pd.read_csv(data_path)

    df.columns = [c.strip().lower() for c in df.columns]

    if "temperature" not in df.columns:
        raise ValueError("Data file must contain a 'temperature' column.")

    temperatures = df["temperature"].to_numpy(dtype=float)

    K_data = {}
    for key in ("K11", "K22", "K33"):
        col = key.lower()
        K_data[key] = df[col].to_numpy(dtype=float) if col in df.columns else None

    available = [k for k, v in K_data.items() if v is not None]
    print(f"Loaded {len(temperatures)} data points.  K directions: {available}")

    return temperatures, K_data


# =============================================================================
# Volume fraction → weight fraction
# =============================================================================

def vf_to_wf(vf: float, rho_f: float, rho_m: float) -> float:
    """
    Convert fiber volume fraction to weight fraction.

        wf = (vf * rho_f) / ((1 - vf) * rho_m + vf * rho_f)

    Parameters
    ----------
    vf, rho_f, rho_m : float

    Returns
    -------
    wf : float
    """
    return (vf * rho_f) / ((1.0 - vf) * rho_m + vf * rho_f)


# =============================================================================
# Output writers
# =============================================================================

def save_csv(temperatures, best_params, K_pred, K_data, output_dir):
    """
    Save estimated constituent and composite conductivities to
    ``estimated_conductivities.csv``.
    """
    poly_model  = PolymerConductivityModel(best_params.p1, best_params.p2)
    fiber_model = FiberConductivityModel(best_params.l1, best_params.l2, best_params.t)
    nan_col     = np.full(len(temperatures), np.nan)

    df = pd.DataFrame({
        "Temperature":   temperatures,
        "K_polymer":     poly_model(temperatures),
        "K_fiber_long":  fiber_model.K_f_long(temperatures),
        "K_fiber_trans": fiber_model.K_f_trans(temperatures),
        "K11_pred":      K_pred[:, 0],
        "K22_pred":      K_pred[:, 1],
        "K33_pred":      K_pred[:, 2],
        "K11_data":      K_data["K11"] if K_data["K11"] is not None else nan_col,
        "K22_data":      K_data["K22"] if K_data["K22"] is not None else nan_col,
        "K33_data":      K_data["K33"] if K_data["K33"] is not None else nan_col,
    })

    out_path = os.path.join(output_dir, "estimated_conductivities.csv")
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved: {out_path}")


def save_plot(temperatures, best_params, K_pred, K_data, output_dir):
    """
    Save ``fit_comparison.png`` — two subplots:
      1. Composite K11/K22/K33: NN prediction vs. measured
      2. Inferred constituent conductivities vs. temperature
    """
    poly_model  = PolymerConductivityModel(best_params.p1, best_params.p2)
    fiber_model = FiberConductivityModel(best_params.l1, best_params.l2, best_params.t)
    T_dense = np.linspace(temperatures.min(), temperatures.max(), 200)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: composite conductivities
    ax  = axes[0]
    cfg = {
        "K11": dict(col=0, color="tab:blue",   label=r"$K_{11}$"),
        "K22": dict(col=1, color="tab:orange",  label=r"$K_{22}$"),
        "K33": dict(col=2, color="tab:green",   label=r"$K_{33}$"),
    }
    for key, c in cfg.items():
        ax.plot(temperatures, K_pred[:, c["col"]],
                color=c["color"], lw=2, label=f"{c['label']} (NN pred.)")
        if K_data.get(key) is not None:
            ax.scatter(temperatures, K_data[key],
                       color=c["color"], marker="o", s=55, zorder=5,
                       label=f"{c['label']} (data)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Thermal conductivity  [W / (m$\cdot$°C)]")
    ax.set_title("Composite conductivities: NN prediction vs. measured")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 2: constituent conductivities
    ax = axes[1]
    ax.plot(T_dense, poly_model(T_dense),
            lw=2, color="tab:red",    label=r"$K_\mathrm{polymer}$")
    ax.plot(T_dense, fiber_model.K_f_long(T_dense),
            lw=2, color="tab:purple", label=r"$K_\mathrm{fiber,\,long}$")
    ax.plot(T_dense, fiber_model.K_f_trans(T_dense),
            lw=2, color="tab:purple", ls="--", label=r"$K_\mathrm{fiber,\,trans}$")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Thermal conductivity  [W / (m$\cdot$°C)]")
    ax.set_title("Inferred constituent conductivities")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, "fit_comparison.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_constituent_plot(temperatures, best_params, output_dir):
    """
    Save ``constituent_conductivities.png`` — three subplots:
      1. Fiber longitudinal conductivity  K_f1  vs. temperature
      2. Fiber transverse conductivity    K_f2  vs. temperature
      3. Matrix (polymer) conductivity    K_m   vs. temperature
    """
    poly_model  = PolymerConductivityModel(best_params.p1, best_params.p2)
    fiber_model = FiberConductivityModel(best_params.l1, best_params.l2, best_params.t)
    T_dense = np.linspace(temperatures.min(), temperatures.max(), 200)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── Subplot 1: fiber longitudinal ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(T_dense, fiber_model.K_f_long(T_dense),
            lw=2.5, color="tab:purple")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Thermal conductivity  [W / (m$\cdot$°C)]")
    ax.set_title(r"Fiber conductivity — direction 1 (longitudinal)  $K_{f,\,1}$")
    ax.grid(True, alpha=0.3)

    # Annotate the slope; near-zero if prior is active
    l1_val = best_params.l1
    ax.text(0.97, 0.05,
            rf"$l_1 = {l1_val:.4g}$ W/(m·°C²)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # ── Subplot 2: fiber transverse ───────────────────────────────────────────
    ax = axes[1]
    ax.plot(T_dense, fiber_model.K_f_trans(T_dense),
            lw=2.5, color="tab:purple", ls="--")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Thermal conductivity  [W / (m$\cdot$°C)]")
    ax.set_title(r"Fiber conductivity — direction 2 (transverse)  $K_{f,\,2}$")
    ax.grid(True, alpha=0.3)

    ax.text(0.97, 0.05,
            rf"$t = {best_params.t:.4g}$  (anisotropy ratio)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # ── Subplot 3: matrix / polymer ───────────────────────────────────────────
    ax = axes[2]
    ax.plot(T_dense, poly_model(T_dense),
            lw=2.5, color="tab:red")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"Thermal conductivity  [W / (m$\cdot$°C)]")
    ax.set_title(r"Matrix (polymer) conductivity  $K_m$")
    ax.grid(True, alpha=0.3)

    ax.text(0.97, 0.05,
            rf"$p_1={best_params.p1:.4g}$, $p_2={best_params.p2:.4g}$ W/(m·°C)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle("Inferred constituent thermal conductivities vs. temperature",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(output_dir, "constituent_conductivities.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_fitted_functions(best_params, output_dir):
    """Save fitted functional forms and parameter values to ``fitted_functions.txt``."""
    out_path = os.path.join(output_dir, "fitted_functions.txt")
    with open(out_path, "w") as f:
        f.write("Fitted constituent thermal conductivity functions\n")
        f.write("=" * 62 + "\n")
        f.write("Reference: Thomas et al. (2024), Comput. Methods Appl. Mech. Eng.\n\n")
        f.write("Polymer (matrix)  — Sec. 4.2:\n")
        f.write(f"  K_polymer(T) = p1 * sqrt(T / 1.0) + p2\n")
        f.write(f"    p1 = {best_params.p1:.8g}  W/(m·°C)\n")
        f.write(f"    p2 = {best_params.p2:.8g}  W/(m·°C)\n\n")
        f.write("Fiber longitudinal (temperature-independent):\n")
        f.write(f"  K_f_long = l2  (constant)\n")
        f.write(f"    l2 = {best_params.l2:.8g}  W/(m·°C)\n\n")
        f.write("Fiber transverse:\n")
        f.write(f"  K_f_trans = l2 / t  (constant)\n")
        f.write(f"    t  = {best_params.t:.8g}  (dimensionless)\n\n")
        f.write("Parameter vector [p1, p2, l2, t]:\n")
        f.write(f"  {best_params.to_array().tolist()}\n")
    print(f"Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inverse estimation of constituent thermal conductivities "
                    "using the trained TC surrogate NN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",  required=True,
                        help="Path to config .py file (e.g. configs/case_1.py)")
    parser.add_argument("--workdir", required=False,
                        help="Directory containing ckpt/ and normalization_stats.npz")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    print(f"\nLoading config  : {args.config}")
    config = load_config(args.config)

    if "inverse_temp" not in config:
        raise ValueError(
            "Config is missing an 'inverse_temp' block.  "
            "Add one following the template in configs/case_1.py."
        )
    it = config.inverse_temp

    # ── Load NN surrogate ─────────────────────────────────────────────────────
    print(f"Loading NN from : {args.workdir}")
    nn_predictor = create_nn_predictor(config, args.workdir)

    # ── Load measurement data ─────────────────────────────────────────────────
    print(f"Loading data    : {it.data}")
    temperatures, K_data = load_data(it.data)

    # ── Build fixed_inputs from config ────────────────────────────────────────
    # vf → w_f conversion (NN trained with weight fraction)
    w_f = vf_to_wf(float(it.vf), float(it.rho_f), float(it.rho_m))
    print(f"\nvf = {it.vf:.4f}  →  w_f = {w_f:.4f}  "
          f"(rho_f = {it.rho_f} kg/m³,  rho_m = {it.rho_m} kg/m³)")

    # Verify orientation tensor sums to 1
    a33_implicit = 1.0 - float(it.a11) - float(it.a22)
    if abs(a33_implicit - float(it.a33)) > 1e-3:
        print(f"Warning: a11+a22+a33 = {float(it.a11)+float(it.a22)+float(it.a33):.4f} != 1.  "
              f"Using a33 = 1 - a11 - a22 = {a33_implicit:.4f}.")

    fixed_inputs = {
        "ar_f":  float(it.aspect_ratio),
        "w_f":   w_f,
        "rho_f": float(it.rho_f),
        "rho_m": float(it.rho_m),
        "a11":   float(it.a11),
        "a22":   float(it.a22),
        "a12":   float(it.a12),
        "a13":   float(it.a13),
        "a23":   float(it.a23),
    }

    print("\n── Fixed structural parameters ──────────────────────────────────────")
    for k, v in fixed_inputs.items():
        print(f"  {k:<8s}: {v}")
    print(f"  {'a33':<8s}: {a33_implicit:.4f}  (implicit = 1 - a11 - a22)")

    # ── Inverse estimation ────────────────────────────────────────────────────
    print(f"\n── Running inverse estimation  ({it.n_restarts} restarts) ─────────")
    best_params, best_loss = run_inverse_estimation(
        temperatures = temperatures,
        K_data       = K_data,
        nn_predictor = nn_predictor,
        fixed_inputs = fixed_inputs,
        n_restarts   = int(it.n_restarts),
        seed         = int(it.seed),
    )

    # ── Console report ────────────────────────────────────────────────────────
    print("\n── Estimated constituent parameters ─────────────────────────────────")
    print(f"  p1  (polymer scaling)          : {best_params.p1:.6g}  W/(m·°C)")
    print(f"  p2  (polymer offset)           : {best_params.p2:.6g}  W/(m·°C)")
    print(f"  l2  (fiber long. conductivity) : {best_params.l2:.6g}  W/(m·°C)  [constant w.r.t. T]")
    print(f"  t   (fiber anisotropy ratio)   : {best_params.t:.6g}")
    print(f"  l1  (fiber temp. slope)        : 0.0  [fixed — fiber T-independent]")
    print(f"\n  Final MSE                      : {best_loss:.6e}  [W/(m·°C)]²")

    K_pred = compute_composite_conductivity(
        best_params, temperatures, nn_predictor, fixed_inputs
    )

    # Per-temperature fit table
    directions = [("K11", 0), ("K22", 1), ("K33", 2)]
    avail      = [(k, c) for k, c in directions if K_data.get(k) is not None]

    print("\n── Per-temperature fit summary ──────────────────────────────────────")
    header = f"{'T [°C]':>8s}"
    for key, _ in avail:
        header += f"  {key}_pred  {key}_data  err%"
    print(header)

    for i, T in enumerate(temperatures):
        row = f"{T:8.1f}"
        for key, col in avail:
            pred = K_pred[i, col]
            data = K_data[key][i]
            pct  = 100.0 * abs(pred - data) / (abs(data) + 1e-12)
            row += f"  {pred:8.4f}  {data:8.4f}  {pct:5.2f}%"
        print(row)

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = str(it.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n── Saving outputs to: {output_dir} ──────────────────────────────────")
    save_csv(temperatures, best_params, K_pred, K_data, output_dir)
    save_plot(temperatures, best_params, K_pred, K_data, output_dir)
    save_constituent_plot(temperatures, best_params, output_dir)
    save_fitted_functions(best_params, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
