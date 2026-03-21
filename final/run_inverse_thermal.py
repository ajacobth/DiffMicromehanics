"""
run_inverse_thermal.py
======================
CLI entry point for temperature-dependent thermal inverse estimation.

Recovers constituent thermal conductivity parameters [p1, p2, l2, t] from
measured composite conductivities (K11, K22, K33) at multiple temperatures.

The problem is described by a JSON file. See thermal_problem.json for schema.

Usage
-----
    python run_inverse_thermal.py                           # uses thermal_problem.json
    python run_inverse_thermal.py --problem my_problem.json
    python run_inverse_thermal.py --problem p.json --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from forward import load_forward
from inverse_thermal import (
    ConstituentParams,
    PolymerConductivityModel,
    FiberConductivityModel,
    make_batched_predictor,
    compute_composite_conductivity,
    run_inverse_estimation,
    vf_to_wf,
)


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(path: str):
    """
    Load temperature and composite conductivity measurements from CSV or Excel.

    Expected columns (case-insensitive): Temperature, K11, K22, K33.
    K22 / K33 are optional — missing columns are silently skipped.

    Returns
    -------
    temperatures : np.ndarray (N,)
    K_data       : dict  'K11'/'K22'/'K33' → np.ndarray (N,) or None
    """
    ext = os.path.splitext(path)[1].lower()
    df  = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "temperature" not in df.columns:
        raise ValueError("Data file must contain a 'temperature' column.")

    temperatures = df["temperature"].to_numpy(dtype=float)
    K_data = {
        key: df[key.lower()].to_numpy(dtype=float) if key.lower() in df.columns else None
        for key in ("K11", "K22", "K33")
    }
    avail = [k for k, v in K_data.items() if v is not None]
    print(f"  Loaded {len(temperatures)} data points.  Available directions: {avail}")
    return temperatures, K_data


# ── output writers ────────────────────────────────────────────────────────────

def save_csv(temperatures, best_params, K_pred, K_data, output_dir):
    poly  = PolymerConductivityModel(best_params.p1, best_params.p2)
    fiber = FiberConductivityModel(best_params.l2, best_params.t)
    nan_c = np.full(len(temperatures), np.nan)

    df = pd.DataFrame({
        "Temperature":   temperatures,
        "K_polymer":     poly(temperatures),
        "K_fiber_long":  fiber.K_f_long(temperatures),
        "K_fiber_trans": fiber.K_f_trans(temperatures),
        "K11_pred":      K_pred[:, 0],
        "K22_pred":      K_pred[:, 1],
        "K33_pred":      K_pred[:, 2],
        "K11_data":      K_data["K11"] if K_data["K11"] is not None else nan_c,
        "K22_data":      K_data["K22"] if K_data["K22"] is not None else nan_c,
        "K33_data":      K_data["K33"] if K_data["K33"] is not None else nan_c,
    })
    path = os.path.join(output_dir, "estimated_conductivities.csv")
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"Saved: {path}")


def save_plots(temperatures, best_params, K_pred, K_data, output_dir):
    poly    = PolymerConductivityModel(best_params.p1, best_params.p2)
    fiber   = FiberConductivityModel(best_params.l2, best_params.t)
    T_dense = np.linspace(temperatures.min(), temperatures.max(), 300)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── subplot 1: composite conductivities ───────────────────────────────────
    ax = axes[0]
    colours = {"K11": "tab:blue", "K22": "tab:orange", "K33": "tab:green"}
    labels  = {"K11": r"$K_{11}$", "K22": r"$K_{22}$", "K33": r"$K_{33}$"}
    for key, col in [("K11", 0), ("K22", 1), ("K33", 2)]:
        c = colours[key];  lb = labels[key]
        ax.plot(temperatures, K_pred[:, col], color=c, lw=2,
                label=f"{lb} (surrogate)")
        if K_data.get(key) is not None:
            ax.scatter(temperatures, K_data[key], color=c, s=55, zorder=5,
                       label=f"{lb} (measured)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Thermal conductivity  [W/(m·°C)]")
    ax.set_title("Composite conductivities: surrogate vs. measured")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── subplot 2: fiber conductivities ───────────────────────────────────────
    ax = axes[1]
    ax.plot(T_dense, fiber.K_f_long(T_dense),
            lw=2.5, color="tab:purple", ls="-",
            label=r"$K_{f,\,\mathrm{long}}$ (longitudinal)")
    ax.plot(T_dense, fiber.K_f_trans(T_dense),
            lw=2.5, color="tab:purple", ls="--",
            label=r"$K_{f,\,\mathrm{trans}}$ (transverse)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Thermal conductivity  [W/(m·°C)]")
    ax.set_title("Fiber thermal conductivity vs. temperature")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.05,
            rf"$l_2={best_params.l2:.4g}$ W/(m·°C)   $t={best_params.t:.4g}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── subplot 3: polymer conductivity ───────────────────────────────────────
    ax = axes[2]
    ax.plot(T_dense, poly(T_dense), lw=2.5, color="tab:red",
            label=r"$K_m(T) = p_1\sqrt{T} + p_2$")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Thermal conductivity  [W/(m·°C)]")
    ax.set_title("Polymer (matrix) thermal conductivity vs. temperature")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.97, 0.05,
            rf"$p_1={best_params.p1:.4g}$,  $p_2={best_params.p2:.4g}$ W/(m·°C)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Thermal inverse estimation results", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "thermal_inverse_results.png")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_summary(best_params, best_loss, output_dir):
    path = os.path.join(output_dir, "thermal_inverse_summary.txt")
    with open(path, "w") as f:
        f.write("Thermal Inverse Estimation — Estimated Parameters\n")
        f.write("=" * 56 + "\n\n")
        f.write("Constituent conductivity models (Thomas et al. 2024):\n\n")
        f.write("  Polymer (matrix):\n")
        f.write(f"    K_m(T) = p1 * sqrt(T / {1.0}) + p2\n")
        f.write(f"    p1 = {best_params.p1:.8g}  W/(m·°C)\n")
        f.write(f"    p2 = {best_params.p2:.8g}  W/(m·°C)\n\n")
        f.write("  Fiber longitudinal (temperature-independent):\n")
        f.write(f"    K_f1 = l2 = {best_params.l2:.8g}  W/(m·°C)\n\n")
        f.write("  Fiber transverse:\n")
        f.write(f"    K_f2 = l2 / t = {best_params.l2 / best_params.t:.8g}  W/(m·°C)\n")
        f.write(f"    t (anisotropy) = {best_params.t:.8g}\n\n")
        f.write(f"  Final MSE = {best_loss:.6e}  [(W/(m·°C))^2]\n")
        f.write(f"\n  Parameter vector [p1, p2, l2, t]: {best_params.to_array().tolist()}\n")
    print(f"Saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(problem_path: str, output_dir_override: Optional[str] = None):
    import json
    with open(problem_path) as f:
        prob = json.load(f)

    fi  = prob["fixed_inputs"]
    vf  = float(fi["vf"])
    rho_f = float(fi["rho_f"])
    rho_m = float(fi["rho_m"])
    w_f   = vf_to_wf(vf, rho_f, rho_m)

    fixed_inputs = {
        "ar_f":  float(fi["ar_f"]),
        "w_f":   w_f,
        "rho_f": rho_f,
        "rho_m": rho_m,
        "a11":   float(fi.get("a11", 0.333)),
        "a22":   float(fi.get("a22", 0.333)),
        "a12":   float(fi.get("a12", 0.0)),
        "a13":   float(fi.get("a13", 0.0)),
        "a23":   float(fi.get("a23", 0.0)),
    }

    n_restarts = int(prob.get("n_restarts", 10))
    seed       = int(prob.get("seed", 0))
    output_dir = output_dir_override or prob.get("output_dir", ".")

    print(f"\nLoading thermal surrogate…")
    fwd       = load_forward("thermal")
    predictor = make_batched_predictor(fwd)

    print(f"Loading data from: {prob['data']}")
    temperatures, K_data = load_data(prob["data"])

    print(f"\nvf={vf:.4f}  →  w_f={w_f:.4f}")
    print("Fixed structural inputs:")
    for k, v in fixed_inputs.items():
        print(f"  {k:<8s}: {v}")

    print(f"\nRunning inverse estimation ({n_restarts} restarts)…")
    best_params, best_loss = run_inverse_estimation(
        temperatures, K_data, predictor, fixed_inputs,
        n_restarts=n_restarts, seed=seed,
        progress_cb=lambda i, n, loss: print(
            f"  Restart {i:2d}/{n}:  loss = {loss:.6e}" +
            (" *best*" if True else "")),
    )

    print("\n=== Estimated constituent parameters ===")
    print(f"  p1  (polymer scaling)   : {best_params.p1:.6g}  W/(m·°C)")
    print(f"  p2  (polymer offset)    : {best_params.p2:.6g}  W/(m·°C)")
    print(f"  l2  (fiber long. K)     : {best_params.l2:.6g}  W/(m·°C)  [constant w.r.t. T]")
    print(f"  t   (anisotropy ratio)  : {best_params.t:.6g}")
    print(f"  Final MSE               : {best_loss:.6e}  [(W/(m·°C))²]")

    K_pred = compute_composite_conductivity(best_params, temperatures, predictor, fixed_inputs)

    avail = [("K11", 0), ("K22", 1), ("K33", 2)]
    avail = [(k, c) for k, c in avail if K_data.get(k) is not None]
    print(f"\n{'T [°C]':>8s}" +
          "".join(f"  {k}_pred  {k}_data  err%" for k, _ in avail))
    for i, T in enumerate(temperatures):
        row = f"{T:8.1f}"
        for key, col in avail:
            pred = K_pred[i, col]
            data = K_data[key][i]
            pct  = 100.0 * abs(pred - data) / (abs(data) + 1e-12)
            row += f"  {pred:7.4f}  {data:7.4f}  {pct:5.2f}%"
        print(row)

    os.makedirs(output_dir, exist_ok=True)
    save_csv(temperatures, best_params, K_pred, K_data, output_dir)
    save_plots(temperatures, best_params, K_pred, K_data, output_dir)
    save_summary(best_params, best_loss, output_dir)
    print("\nDone.")
    return best_params, best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thermal inverse estimation (temperature-dependent constituent K).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--problem",    default="thermal_problem.json",
                        help="Path to problem JSON file")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory (default: from JSON)")
    args = parser.parse_args()
    run(args.problem, args.output_dir)
