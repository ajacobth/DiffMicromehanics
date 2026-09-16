"""
Plot LSAM thermal conductivity: model prediction (card 16) vs experimental data.
Edit the USER SETTINGS block below to adjust figure preferences.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from core.services.service_transfer import run_thermal_sweep

# ── USER SETTINGS ─────────────────────────────────────────────────────────────
SAVE_PATH   = "figures/lsam_thermal_conductivity.png"   # output path (relative to final/)
FIGSIZE     = (12, 4)        # (width, height) in inches
DPI         = 300
FONT_SIZE   = 11             # base font size
LINE_WIDTH  = 1.8
MARKER_SIZE = 4
ALPHA_EXP   = 0.55           # transparency of experimental scatter points
T_MIN, T_MAX = 20, 195       # temperature axis limits (°C)
Y_LIMS      = {              # y-axis limits per channel (None = auto)
    "K11": (1.0, 2.0),
    "K22": (0.3, 1.0),
    "K33": (0.2, 0.6),
}
COLORS = {
    "pred":        "#1f77b4",   # model line
    "large":       "#d62728",   # large sample scatter
    "small":       "#ff7f0e",   # small sample scatter
    "edge":        "#d62728",   # edge scatter (K22)
    "middle":      "#ff7f0e",   # middle scatter (K22)
}
CHANNEL_LABELS = {"K11": r"$K_{11}$", "K22": r"$K_{22}$", "K33": r"$K_{33}$"}
# ── END USER SETTINGS ─────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "data", "PESU_LSAM_DATA_TC.xlsx")
CARD_ID   = 16
MICRO     = {"mf": 0.25, "ar": 20.0, "w_f": 0.25,
             "a11": 0.6507, "a22": 0.2637, "a12": 0.0, "a13": 0.0, "a23": 0.0}

def read_sheet(xl, name):
    """Read a sheet where the first row is data, not a header."""
    df = xl.parse(name, header=None)
    df.columns = ["T", "K"]
    df = df.dropna().astype(float)
    return df[df["T"].between(T_MIN, T_MAX)]

def load_exp():
    xl = pd.ExcelFile(DATA_PATH)
    return {
        "K11_large":  read_sheet(xl, "LSAM_LARGE_K11"),
        "K11_small":  read_sheet(xl, "LSAM_SMALL_K11"),
        "K22_edge":   read_sheet(xl, "LSAM_EDGE_K22"),
        "K22_middle": read_sheet(xl, "LSAM_MIDDLE_K22"),
        "K33_large":  read_sheet(xl, "LSAM_LARGE_K33"),
        "K33_small":  read_sheet(xl, "LSAM_SMALL_K33"),
    }

def compute_errors(pred_T, pred_K, exp_df):
    """Interpolate prediction at experimental T values, return residuals (%)."""
    errors = []
    for _, row in exp_df.iterrows():
        k_pred = np.interp(row["T"], pred_T, pred_K)
        errors.append(100.0 * (k_pred - row["K"]) / row["K"])
    return np.array(errors)

def main():
    mpl.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
                         "xtick.labelsize": FONT_SIZE - 1, "ytick.labelsize": FONT_SIZE - 1,
                         "legend.fontsize": FONT_SIZE - 1, "figure.dpi": DPI,
                         "axes.spines.top": False, "axes.spines.right": False})

    # ── Model prediction ──────────────────────────────────────────────────────
    pred_temps = list(np.linspace(T_MIN, T_MAX, 80))
    sweep = run_thermal_sweep(CARD_ID, MICRO, pred_temps)
    T_pred = np.array(sweep["temperatures"])
    K_pred = {"K11": np.array(sweep["K11"]),
              "K22": np.array(sweep["K22"]),
              "K33": np.array(sweep["K33"])}

    # ── Experimental data ─────────────────────────────────────────────────────
    exp = load_exp()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=False, constrained_layout=True)
    configs = [
        ("K11", "K11_large", "K11_small", "Large sample", "Small sample", COLORS["large"], COLORS["small"]),
        ("K22", "K22_edge",  "K22_middle","Edge",         "Middle",       COLORS["edge"],  COLORS["middle"]),
        ("K33", "K33_large", "K33_small", "Large sample", "Small sample", COLORS["large"], COLORS["small"]),
    ]

    print(f"\n{'Channel':<6} {'Sample':<12} {'N':>4} {'Mean err%':>10} {'Std err%':>9} {'Max|err|%':>10}")
    print("-" * 55)

    for ax, (ch, key1, key2, lbl1, lbl2, c1, c2) in zip(axes, configs):
        # model line
        ax.plot(T_pred, K_pred[ch], color=COLORS["pred"], lw=LINE_WIDTH, label="Model", zorder=3)

        # scatter
        for key, lbl, c in [(key1, lbl1, c1), (key2, lbl2, c2)]:
            df = exp[key]
            ax.scatter(df["T"], df["K"], s=MARKER_SIZE**2, color=c, alpha=ALPHA_EXP,
                       label=lbl, zorder=2)
            errs = compute_errors(T_pred, K_pred[ch], df)
            print(f"{ch:<6} {lbl:<12} {len(errs):>4} {errs.mean():>+10.2f} {errs.std():>9.2f} {np.abs(errs).max():>10.2f}")

        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel(f"{CHANNEL_LABELS[ch]} (W/m·K)")
        ax.set_xlim(T_MIN, T_MAX)
        if Y_LIMS[ch]:
            ax.set_ylim(*Y_LIMS[ch])
        ax.legend(frameon=False)

    out = os.path.join(os.path.dirname(__file__), "..", SAVE_PATH)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"\nFigure saved → {os.path.abspath(out)}")
    pdf_out = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(pdf_out, bbox_inches="tight")
    print(f"Figure saved → {os.path.abspath(pdf_out)}")
    plt.show()

if __name__ == "__main__":
    main()
