"""
Created on Wed Jul 10 17:10:00 2024

@author: akshayjacobthomas
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import ml_collections
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from NN_surrogate.utils import restore_checkpoint
import models

plt.rcParams["text.usetex"] = True


# ============================================================
# Output naming (given)
# ============================================================
OUTPUT_FIELD_NAMES = [
    "k11", "k12", "k13", "k22", "k23", "k33"
]

# ============================================================
# Units displayed on plots (TARGET display units)
# ============================================================
OUTPUT_UNITS: Dict[str, Optional[str]] = {
    "k11": r"W/(m$\cdot$K)", "k12": r"W/(m$\cdot$K)", "k13": r"W/(m$\cdot$K)",
    "k22": r"W/(m$\cdot$K)", "k23": r"W/(m$\cdot$K)", "k33": r"W/(m$\cdot$K)",
}

# ============================================================
# Pretty display names for axes/titles
# ============================================================
OUTPUT_DISPLAY_NAMES: Dict[str, str] = {
    "k11": r"$k_{11}$", "k12": r"$k_{12}$", "k13": r"$k_{13}$",
    "k22": r"$k_{22}$", "k23": r"$k_{23}$", "k33": r"$k_{33}$",
}

# ============================================================
# Scaling factors (raw -> TARGET units in OUTPUT_UNITS)
# ============================================================
OUTPUT_SCALES: Dict[str, float] = {
    "k11": 1.0, "k12": 1.0, "k13": 1.0,
    "k22": 1.0, "k23": 1.0, "k33": 1.0,
}


# ============================================================
# Output metadata
# ============================================================
@dataclass
class OutputMeta:
    display_name: str
    raw_name: str
    unit: Optional[str] = None
    scale: float = 1.0
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_numpy(x) -> np.ndarray:
    return np.asarray(x)


def make_label(meta: OutputMeta) -> str:
    if meta.unit:
        return f"{meta.display_name}[{meta.unit}]"
    return meta.display_name


def savefig(path: str, dpi: int = 220):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def maybe_downsample(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    idx = np.arange(n)
    if n <= max_points:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


# ============================================================
# Metrics
# ============================================================
def per_output_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    eps = 1e-12

    resid = y_pred - y_true
    mse = np.mean(resid**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(resid), axis=0)

    ss_res = np.sum(resid**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True))**2, axis=0) + eps
    r2 = 1.0 - ss_res / ss_tot

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = to_numpy(y_true).reshape(-1)
    y_pred = to_numpy(y_pred).reshape(-1)
    eps = 1e-12

    resid = y_pred - y_true
    mse = float(np.mean(resid**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(resid)))

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2)) + eps
    r2 = float(1.0 - ss_res / ss_tot)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# ============================================================
# Plots
# ============================================================

def plot_parity_hexbin_pcterr(
    y_true_1d: np.ndarray,
    y_pred_1d: np.ndarray,
    meta: OutputMeta,
    out_path: str,
    gridsize: int = 30,
    eps_denom: float = 1e-12,
    clip_pct: Optional[float] = 200.0,
):
    """
    Parity plot with hexbin coloring by MEAN Absolute Error per bin.

    clip_pct: cap absolute error to reduce colorbar domination by extreme outliers.
              Set None to disable clipping.
    """
    yt = to_numpy(y_true_1d).astype(float)
    yp = to_numpy(y_pred_1d).astype(float)

    ape = np.abs(yp - yt)  # absolute error

    if clip_pct is not None:
        ape = np.clip(ape, 0.0, float(clip_pct))

    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))

    plt.figure(figsize=(5.8, 5.3))
    hb = plt.hexbin(
        yt, yp,
        C=ape,
        reduce_C_function=np.mean,  # mean absolute error per cell
        gridsize=gridsize,
        mincnt=1
    )
    plt.plot([lo, hi], [lo, hi], linewidth=2)

    plt.xlabel(f"Actual {make_label(meta)}")
    plt.ylabel(f"Predicted {make_label(meta)}")
    plt.title(f"Parity Plot: {make_label(meta)}")

    cbar = plt.colorbar(hb)
    if clip_pct is None:
        cbar.set_label("Mean Absolute Error")
    else:
        cbar.set_label("Mean Absolute Error")

    if meta.xlim is not None:
        plt.xlim(meta.xlim)
    if meta.ylim is not None:
        plt.ylim(meta.ylim)

    savefig(out_path)


def plot_residuals_vs_actual(
    y_true_1d: np.ndarray,
    y_pred_1d: np.ndarray,
    meta: OutputMeta,
    out_path: str,
    max_points: int = 800000,
):
    yt = to_numpy(y_true_1d)
    yp = to_numpy(y_pred_1d)
    resid = yp - yt

    idx = maybe_downsample(len(yt), max_points=max_points, seed=1)

    plt.figure(figsize=(6.4, 4.6))
    plt.scatter(yt[idx], resid[idx], s=10, alpha=0.5, edgecolor="none")
    plt.axhline(0.0, linewidth=2)
    plt.xlabel(f"Actual {make_label(meta)}")
    plt.ylabel(r"Residual $(\hat{y}-y)$")
    plt.title(f"Residuals vs Actual: {make_label(meta)}")

    if meta.xlim is not None:
        plt.xlim(meta.xlim)

    savefig(out_path)


def plot_residual_hist(
    y_true_1d: np.ndarray,
    y_pred_1d: np.ndarray,
    meta: OutputMeta,
    out_path: str,
    bins: int = 60,
):
    yt = to_numpy(y_true_1d)
    yp = to_numpy(y_pred_1d)
    resid = yp - yt

    plt.figure(figsize=(6.4, 4.2))
    plt.hist(resid, bins=bins, edgecolor="black", alpha=0.85)
    plt.axvline(0.0, linewidth=2)
    plt.xlabel(rf"Residual $(\hat{{y}}-y)$ for {make_label(meta)}")
    plt.ylabel("Count")
    plt.title(f"Residual histogram: {make_label(meta)}")
    savefig(out_path)


def plot_abs_error_cdf(
    y_true_1d: np.ndarray,
    y_pred_1d: np.ndarray,
    meta: OutputMeta,
    out_path: str,
):
    yt = to_numpy(y_true_1d)
    yp = to_numpy(y_pred_1d)
    abs_err = np.abs(yp - yt)

    abs_sorted = np.sort(abs_err)
    cdf = np.arange(1, len(abs_sorted) + 1) / len(abs_sorted)

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(abs_sorted, cdf, linewidth=2)
    plt.xlabel(rf"Absolute error $|\hat{{y}}-y|$ for {make_label(meta)}")
    plt.ylabel("CDF")
    plt.title(f"Absolute error CDF: {make_label(meta)}")
    savefig(out_path)


def plot_per_output_bars(metrics: Dict[str, np.ndarray], metas: List[OutputMeta], out_dir: str):
    names = [m.display_name for m in metas]
    x = np.arange(len(metas))

    plt.figure(figsize=(9.0, 4.2))
    plt.bar(x, metrics["RMSE"])
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("RMSE")
    plt.title("RMSE per output")
    savefig(os.path.join(out_dir, "rmse_per_output.png"))

    plt.figure(figsize=(9.0, 4.2))
    plt.bar(x, metrics["MAE"])
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("MAE")
    plt.title("MAE per output")
    savefig(os.path.join(out_dir, "mae_per_output.png"))

    plt.figure(figsize=(9.0, 4.2))
    plt.bar(x, metrics["R2"])
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel(r"$R^2$")
    plt.title(r"$R^2$ per output")
    savefig(os.path.join(out_dir, "r2_per_output.png"))


def save_worst_cases_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metas: List[OutputMeta],
    out_path: str,
    k: int = 15,
):
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)
    abs_err = np.abs(y_pred - y_true)

    worst = np.argsort(abs_err.max(axis=1))[::-1][:k]

    with open(out_path, "w") as f:
        f.write(f"Top-{k} worst samples by max abs error across outputs\n")
        f.write("=" * 72 + "\n\n")
        for rank, i in enumerate(worst, start=1):
            f.write(f"[{rank}] idx={i}\n")
            f.write(f"  max|err|  = {abs_err[i].max():.6g}\n")
            f.write(f"  mean|err| = {abs_err[i].mean():.6g}\n")
            for j, meta in enumerate(metas):
                f.write(
                    f"    - {meta.raw_name} ({make_label(meta)}): "
                    f"true={y_true[i, j]:.6g}, pred={y_pred[i, j]:.6g}, abs_err={abs_err[i, j]:.6g}\n"
                )
            f.write("\n")


# ============================================================
# Evaluate (called from your other script)
# ============================================================
def evaluate(config: ml_collections.ConfigDict, workdir: str):
    model = models.MICRO_SURROGATE_L2(config)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    state = restore_checkpoint(model.state, ckpt_path)
    params = state["params"]

    test_data = np.genfromtxt("TCPSD_dataset_test.csv", delimiter=",", skip_header=1)
    input_dim = int(config.input_dim)
    output_dim = int(config.output_dim)

    test_inputs = test_data[:, :input_dim]
    test_targets = test_data[:, input_dim:]

    if test_targets.shape[1] != output_dim:
        raise ValueError(f"Expected output_dim={output_dim}, got {test_targets.shape[1]}")

    if len(OUTPUT_FIELD_NAMES) != output_dim:
        raise ValueError(
            f"OUTPUT_FIELD_NAMES length={len(OUTPUT_FIELD_NAMES)} but output_dim={output_dim}.\n"
            f"Fix OUTPUT_FIELD_NAMES or config.output_dim."
        )

    norm_stats_path = os.path.join(workdir, "normalization_stats.npz")
    norm_stats = np.load(norm_stats_path)

    input_mean = jnp.array(norm_stats["input_mean"])
    input_std  = jnp.array(norm_stats["input_std"])
    target_mean = jnp.array(norm_stats["target_mean"])
    target_std  = jnp.array(norm_stats["target_std"])

    test_inputs_norm = (test_inputs - input_mean) / input_std

    test_preds = model.u_net(params, test_inputs_norm)
    test_preds = (test_preds * target_std) + target_mean

    y_pred = np.array(test_preds)
    y_true = np.array(test_targets)

    metas: List[OutputMeta] = []
    for raw_name in OUTPUT_FIELD_NAMES:
        unit = OUTPUT_UNITS.get(raw_name, None)
        scale = float(OUTPUT_SCALES.get(raw_name, 1.0))
        display_name = OUTPUT_DISPLAY_NAMES.get(raw_name, raw_name)
        metas.append(OutputMeta(display_name=display_name, raw_name=raw_name, unit=unit, scale=scale))

    scales = np.array([m.scale for m in metas], dtype=float).reshape(1, -1)
    y_true_s = y_true * scales
    y_pred_s = y_pred * scales

    g = global_metrics(y_true_s, y_pred_s)
    m = per_output_metrics(y_true_s, y_pred_s)

    print("Evaluation Results (in displayed units):")
    print(f"MSE  : {g['MSE']:.6f}")
    print(f"RMSE : {g['RMSE']:.6f}")
    print(f"MAE  : {g['MAE']:.6f}")
    print(f"R^2  : {g['R2']:.6f}")
    for j, meta in enumerate(metas):
        print(
            f"[{j:02d}] {meta.raw_name:5s} -> {make_label(meta):30s}  "
            f"RMSE={m['RMSE'][j]:.6g}  MAE={m['MAE'][j]:.6g}  R2={m['R2'][j]:.6g}"
        )

    run_name = str(config.wandb.name)
    out_dir = ensure_dir(os.path.join(workdir, "eval_figs", run_name))
    plots_dir = ensure_dir(os.path.join(out_dir, "plots"))

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write("Global metrics (in displayed units)\n")
        f.write(f"MSE  : {g['MSE']:.8g}\n")
        f.write(f"RMSE : {g['RMSE']:.8g}\n")
        f.write(f"MAE  : {g['MAE']:.8g}\n")
        f.write(f"R2   : {g['R2']:.8g}\n\n")
        f.write("Per-output metrics (in displayed units)\n")
        for j, meta in enumerate(metas):
            f.write(
                f"[{j:02d}] {meta.raw_name:5s} ({make_label(meta)})  "
                f"MSE={m['MSE'][j]:.8g}  RMSE={m['RMSE'][j]:.8g}  "
                f"MAE={m['MAE'][j]:.8g}  R2={m['R2'][j]:.8g}\n"
            )

    for j, meta in enumerate(metas):
        yt = y_true_s[:, j]
        yp = y_pred_s[:, j]

        plot_parity_hexbin_pcterr(
            yt, yp, meta,
            out_path=os.path.join(plots_dir, f"parity_hexbin_pcterr_{meta.raw_name}.png"),
            gridsize=55,
            eps_denom=1e-12,
            clip_pct=200.0,
        )

        plot_residuals_vs_actual(
            yt, yp, meta,
            out_path=os.path.join(plots_dir, f"residuals_vs_actual_{meta.raw_name}.png"),
            max_points=80000,
        )

        plot_residual_hist(
            yt, yp, meta,
            out_path=os.path.join(plots_dir, f"residual_hist_{meta.raw_name}.png"),
            bins=60,
        )

        plot_abs_error_cdf(
            yt, yp, meta,
            out_path=os.path.join(plots_dir, f"abs_error_cdf_{meta.raw_name}.png"),
        )

    plot_per_output_bars(m, metas, out_dir=out_dir)

    save_worst_cases_report(
        y_true_s, y_pred_s, metas,
        out_path=os.path.join(out_dir, "worst_cases.txt"),
        k=15,
    )

    print(f"\nSaved evaluation artifacts to:\n  {out_dir}")

    return {"global": g, "per_output": m}
