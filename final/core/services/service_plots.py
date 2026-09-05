"""
Plotting helpers for the DiffMicromechanics agent.

Public functions:
    plot_k_vs_T      — K11/K22/K33 vs temperature, optional experimental overlay
    plot_sensitivity  — tornado chart: sensitivity of one composite property
                        to each free microstructure/constituent parameter
"""

from __future__ import annotations

import base64
import io
import tempfile
import webbrowser
from typing import Optional

import numpy as np

# Use Agg so plots render without a display (agent is headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── NeurIPS-style constants ───────────────────────────────────────────────────

_W, _H   = 3.5, 2.6                          # single-column figure (inches)
_FONT    = 8
_LWIDTH  = 1.1
_MSIZE   = 3.5
_EWIDTH  = 0.7
_COLORS  = ["#2166ac", "#d6604d", "#4dac26"]  # blue / red / green

_CHANNEL_STYLE = {
    "k11": dict(color=_COLORS[0], ls="-",  label="K₁₁ (axial)"),
    "k22": dict(color=_COLORS[1], ls="--", label="K₂₂ (transverse)"),
    "k33": dict(color=_COLORS[2], ls=":",  label="K₃₃ (out-of-plane)"),
}

_PARAM_LABELS = {
    "a11":            "a₁₁ (alignment)",
    "a22":            "a₂₂ (transverse)",
    "fiber_massfrac": "wf (mass fraction)",
    "ar":             "AR (aspect ratio)",
    "matrix_modulus": "Em (matrix modulus)",
    "matrix_poisson": "νm (matrix Poisson)",
    "f_cte1":         "αf₁ (fiber CTE axial)",
    "f_cte2":         "αf₂ (fiber CTE transverse)",
    "m_cte":          "αm (matrix CTE)",
}

# Sweep ranges for sensitivity: (lo_frac, hi_frac) relative to card base value
# Falls back to ±25% when not listed
_SWEEP_FRAC = {
    "a11":            (0.80, 1.20),
    "a22":            (0.50, 1.60),
    "fiber_massfrac": (0.70, 1.30),
    "ar":             (0.50, 2.00),
    "matrix_modulus": (0.70, 1.30),
    "matrix_poisson": (0.85, 1.15),
    "f_cte1":         (2.0,  -2.0),   # absolute — see below
    "f_cte2":         (0.70, 1.30),
    "m_cte":          (0.70, 1.30),
}
_N_SWEEP = 12    # points per parameter


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _open_in_browser(fig: plt.Figure, title: str = "DiffMicromechanics Plot") -> None:
    """Render figure to base64 PNG, embed in minimal HTML, open in browser."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()

    html = (
        "<!doctype html><html><head>"
        f"<title>{title}</title>"
        "<style>body{margin:0;background:#fff;display:flex;"
        "justify-content:center;align-items:flex-start;padding:24px}"
        "img{max-width:100%;height:auto;box-shadow:0 1px 4px rgba(0,0,0,.15)}"
        "</style></head><body>"
        f'<img src="data:image/png;base64,{b64}">'
        "</body></html>"
    )

    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(html)
        path = f.name

    webbrowser.open(f"file://{path}")


# ── Thermal conductivity plot ─────────────────────────────────────────────────

def plot_k_vs_T(
    card_inputs:        dict,
    exp_data:           Optional[dict] = None,
    micro_overrides:    Optional[dict] = None,
) -> str:
    """
    Plot composite K11/K22/K33 vs temperature from the parametric Stage 3 model.

    Parameters
    ----------
    card_inputs      : dict from load_card_inputs — must contain p1, p2, k_f1, k_f2
                       plus structural fields (a11, a22, ar, fiber_massfrac, rho_f, rho_m)
    exp_data         : optional dict from _parse_thermal_csv — {'K11': {'T': [...], 'K': [...]}, ...}
    micro_overrides  : optional dict with a11/a22/ar/fiber_massfrac overrides (post-transfer)

    Returns
    -------
    Path to the saved PNG file.
    """
    from core.services.service_forward import get_model
    from core.inverse_thermal import (
        ConstituentParams, compute_composite_conductivity, make_batched_predictor,
    )

    # Merge microstructure overrides
    inputs = dict(card_inputs)
    if micro_overrides:
        inputs.update(micro_overrides)
        if "fiber_massfrac" in micro_overrides and "w_f" not in micro_overrides:
            inputs["w_f"] = micro_overrides["fiber_massfrac"]
        if "ar" in micro_overrides and "ar_f" not in micro_overrides:
            inputs["ar_f"] = micro_overrides["ar"]

    for src, dst in [("fiber_density", "rho_f"), ("matrix_density", "rho_m"),
                     ("fiber_massfrac", "w_f"), ("ar", "ar_f")]:
        if dst not in inputs and src in inputs:
            inputs[dst] = inputs[src]

    has_parametric = all(inputs.get(k) is not None for k in ("p1", "p2", "k_f1", "k_f2"))
    if not has_parametric:
        raise ValueError("Stage 3 parameters (p1, p2, k_f1, k_f2) not found in card inputs.")

    required_struct = ["ar_f", "w_f", "rho_f", "rho_m", "a11", "a22", "a12", "a13", "a23"]
    for k in required_struct:
        if k not in inputs:
            inputs.setdefault(k, 0.0)

    p1, p2   = float(inputs["p1"]), float(inputs["p2"])
    k_f1     = float(inputs["k_f1"])
    k_f2     = float(inputs["k_f2"])
    t        = k_f1 / k_f2 if k_f2 > 0 else 1.01
    params   = ConstituentParams(p1=p1, p2=p2, l2=k_f1, t=t)

    fixed = {k: float(inputs[k]) for k in required_struct}

    T_model  = np.linspace(25.0, 200.0, 60)
    predictor = make_batched_predictor(get_model("thermal"))
    K_pred    = compute_composite_conductivity(params, T_model, predictor, fixed)

    fig, ax = plt.subplots(figsize=(_W, _H))

    for i, (key, style) in enumerate(_CHANNEL_STYLE.items()):
        col_idx = {"k11": 0, "k22": 1, "k33": 2}[key]
        ax.plot(T_model, K_pred[:, col_idx],
                color=style["color"], ls=style["ls"],
                lw=_LWIDTH, label=style["label"])

        if exp_data:
            ch_key = key.upper()
            ch = exp_data.get(ch_key)
            if ch is not None:
                ax.scatter(ch["T"], ch["K"],
                           color=style["color"], s=_MSIZE ** 2,
                           zorder=5, linewidths=_EWIDTH,
                           edgecolors="white")

    ax.set_xlabel("Temperature  (°C)", fontsize=_FONT)
    ax.set_ylabel("Thermal conductivity  (W/m·K)", fontsize=_FONT)
    ax.tick_params(labelsize=_FONT - 1)
    ax.legend(fontsize=_FONT - 1, frameon=False, loc="upper right")
    ax.grid(axis="y", lw=0.4, alpha=0.4, color="grey")
    _despine(ax)
    fig.tight_layout(pad=0.6)

    _open_in_browser(fig, "Thermal Conductivity vs Temperature")
    return "opened"


# ── Sensitivity tornado plot ──────────────────────────────────────────────────

def plot_sensitivity(
    card_inputs:      dict,
    target_property:  str,
    free_variables:   list[str],
) -> str:
    """
    Tornado plot: sensitivity of `target_property` to each parameter in
    `free_variables`, evaluated by sweeping each across its typical range
    while keeping others at their card base values.

    Parameters
    ----------
    card_inputs      : dict from load_card_inputs (base point)
    target_property  : output name, e.g. "E1", "CTE11", "nu12"
    free_variables   : list of input parameter names to analyse

    Returns
    -------
    Path to the saved PNG file.
    """
    from core.services.service_forward import run_forward, get_output_fields

    # Determine which model to run
    _TE_OUTPUTS = {"CTE11", "CTE22", "CTE33", "CTE12", "CTE13", "CTE23"}
    model_name  = "thermoelastic" if target_property in _TE_OUTPUTS else "elastic"

    # Canonicalise model inputs (strip non-model fields)
    from core.services.service_forward import get_input_fields
    model_fields = set(get_input_fields(model_name))

    base = {k: float(v) for k, v in card_inputs.items()
            if k in model_fields and v is not None}

    # Base prediction
    try:
        base_pred  = run_forward(model_name, base)
        base_value = base_pred.get(target_property)
        if base_value is None:
            raise ValueError(f"{target_property} not in model outputs")
        base_value = float(base_value)
    except Exception as e:
        raise ValueError(f"Base prediction failed: {e}")

    # Sweep each parameter
    bars: list[tuple[str, float, float]] = []   # (param, lo_pct, hi_pct)

    for param in free_variables:
        if param not in base:
            continue
        base_val = base[param]
        if base_val == 0.0:
            continue

        lo_f, hi_f = _SWEEP_FRAC.get(param, (0.75, 1.25))

        # f_cte1 can be negative — use absolute shifts instead
        if param == "f_cte1":
            lo_val = base_val - abs(base_val) * 1.0
            hi_val = base_val + abs(base_val) * 1.0
        else:
            lo_val = base_val * lo_f
            hi_val = base_val * hi_f

        lo_val, hi_val = min(lo_val, hi_val), max(lo_val, hi_val)
        sweep_vals = np.linspace(lo_val, hi_val, _N_SWEEP)

        preds = []
        for v in sweep_vals:
            inp = {**base, param: float(v)}
            try:
                out = run_forward(model_name, inp)
                preds.append(float(out.get(target_property, base_value)))
            except Exception:
                preds.append(base_value)

        lo_pct = (min(preds) - base_value) / abs(base_value) * 100.0
        hi_pct = (max(preds) - base_value) / abs(base_value) * 100.0

        # Sort so lo is always the "left" end (may differ sign from above)
        bars.append((param, lo_pct, hi_pct))

    if not bars:
        raise ValueError("No valid parameters to sweep.")

    # Sort by total range (|hi - lo|), largest at top
    bars.sort(key=lambda b: abs(b[2] - b[1]), reverse=True)

    params  = [b[0] for b in bars]
    lo_vals = [b[1] for b in bars]
    hi_vals = [b[2] for b in bars]
    labels  = [_PARAM_LABELS.get(p, p) for p in params]

    n    = len(bars)
    h    = max(2.0, 0.35 * n + 0.6)
    fig, ax = plt.subplots(figsize=(_W, h))

    y_pos = np.arange(n)

    for i, (lo, hi) in enumerate(zip(lo_vals, hi_vals)):
        # Left bar (negative or lower side)
        left  = min(lo, 0.0)
        right = max(hi, 0.0)
        neg   = min(lo, hi, 0.0)
        pos   = max(lo, hi, 0.0)

        if neg < 0:
            ax.barh(i, neg, left=0, height=0.55,
                    color=_COLORS[1], alpha=0.85)
        if pos > 0:
            ax.barh(i, pos, left=0, height=0.55,
                    color=_COLORS[0], alpha=0.85)

    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=_FONT - 1)
    ax.set_xlabel(f"Δ {target_property} / base  (%)", fontsize=_FONT)
    ax.tick_params(axis="x", labelsize=_FONT - 1)
    ax.invert_yaxis()
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", lw=0.4, alpha=0.4, color="grey")
    fig.tight_layout(pad=0.6)

    _open_in_browser(fig, f"Sensitivity — {target_property}")
    return "opened"


# ── Sweep line plot ───────────────────────────────────────────────────────────

_PARAM_AXIS_LABELS = {
    "fiber_massfrac": "Fiber mass fraction  wf",
    "a11":            "Orientation tensor  a₁₁",
    "a22":            "Orientation tensor  a₂₂",
    "ar":             "Aspect ratio  AR",
    "matrix_modulus": "Matrix modulus  Em  (MPa)",
    "matrix_poisson": "Matrix Poisson ratio  νm",
    "f_cte1":         "Fiber axial CTE  αf₁  (1/K)",
    "f_cte2":         "Fiber transverse CTE  αf₂  (1/K)",
    "m_cte":          "Matrix CTE  αm  (1/K)",
}

_PROP_LINESTYLES = ["-", "--", ":", "-."]

_T_REF_SWEEP = 1.0   # must match agent_tools._T_REF

_THERMAL_OUTPUTS = {"K11", "K22", "K33"}
_THERMAL_STRUCT  = ["ar_f", "w_f", "rho_f", "rho_m", "a11", "a22", "a12", "a13", "a23"]
_SWEEP_ALIASES   = {"ar": "ar_f", "fiber_massfrac": "w_f"}


def plot_sweep(
    card_inputs:     dict,
    parameter:       str,
    target_props:    list[str],
    x_min:           float,
    x_max:           float,
    n_points:        int = 10,
    temperature_C:   float = 25.0,
) -> str:
    """
    Line plot: one or more composite properties vs a swept parameter.

    Parameters
    ----------
    card_inputs   : dict from load_card_inputs (base point)
    parameter     : the input to vary (e.g. "ar", "fiber_massfrac", "a11")
    target_props  : list of output names to plot — elastic (E1/E2/E3/G12/nu12),
                    thermoelastic (CTE11/CTE22), or thermal (K11/K22/K33)
    x_min, x_max  : sweep range for the parameter
    n_points      : number of evenly spaced points (default 10)
    temperature_C : fixed temperature for thermal sweeps (default 25°C)

    Returns
    -------
    Numerical data table as a string (also opens chart in browser).
    """
    from core.services.service_forward import run_forward, get_input_fields

    _TE_OUTPUTS = {"CTE11", "CTE22", "CTE33", "CTE12", "CTE13", "CTE23"}

    sweep_vals = np.linspace(x_min, x_max, n_points)
    needs_thermal = any(p in _THERMAL_OUTPUTS for p in target_props)

    results: dict[str, list[float]] = {p: [] for p in target_props}

    if needs_thermal:
        # ── Thermal sweep ──────────────────────────────────────────────────────
        inputs = dict(card_inputs)
        # Resolve field-name aliases
        for src, dst in [("fiber_density", "rho_f"), ("matrix_density", "rho_m"),
                         ("fiber_massfrac", "w_f"), ("ar", "ar_f")]:
            if dst not in inputs and src in inputs:
                inputs[dst] = inputs[src]

        # Compute k_m at the fixed temperature from parametric model or scalar
        p1   = inputs.get("p1")
        p2   = inputs.get("p2")
        k_f1 = inputs.get("k_f1")
        k_f2 = inputs.get("k_f2")
        k_m_scalar = inputs.get("k_m")

        if all(v is not None for v in (p1, p2, k_f1, k_f2)):
            k_m_base = float(p1) * (max(float(temperature_C), 0.0) / _T_REF_SWEEP) ** 0.5 + float(p2)
        elif k_m_scalar is not None:
            k_m_base = float(k_m_scalar)
        else:
            raise ValueError(
                "No thermal conductivity data in inputs. "
                "Provide k_f1, k_f2 and either p1+p2 (parametric) or k_m (scalar)."
            )

        base_thermal = {
            "k_f1": float(k_f1),
            "k_f2": float(k_f2),
            "k_m":  k_m_base,
        }
        for field in _THERMAL_STRUCT:
            if field in inputs:
                base_thermal[field] = float(inputs[field])
            else:
                base_thermal[field] = 0.0

        _THERMAL_KEY_MAP = {"K11": "k11", "K22": "k22", "K33": "k33"}

        for v in sweep_vals:
            tinp = dict(base_thermal)
            tinp[parameter] = float(v)
            alias = _SWEEP_ALIASES.get(parameter)
            if alias:
                tinp[alias] = float(v)
            try:
                out = run_forward("thermal", tinp)
                for p in target_props:
                    model_key = _THERMAL_KEY_MAP.get(p, p.lower())
                    results[p].append(float(out.get(model_key, float("nan"))))
            except Exception:
                for p in target_props:
                    results[p].append(float("nan"))

        y_label = f"Thermal conductivity  (W/m·K)  at T={temperature_C:.0f}°C"
        y_scale = 1.0

    else:
        # ── Elastic / thermoelastic sweep ──────────────────────────────────────
        needs_te   = any(p in _TE_OUTPUTS for p in target_props)
        model_name = "thermoelastic" if needs_te else "elastic"
        model_fields = set(get_input_fields(model_name))
        base = {k: float(v) for k, v in card_inputs.items()
                if k in model_fields and v is not None}

        for v in sweep_vals:
            inp = {**base, parameter: float(v)}
            try:
                out = run_forward(model_name, inp)
                for p in target_props:
                    results[p].append(float(out.get(p, float("nan"))))
            except Exception:
                for p in target_props:
                    results[p].append(float("nan"))

        _MODULI = {"E1", "E2", "E3", "G12", "G13", "G23"}
        _CTE    = {"CTE11", "CTE22", "CTE33"}
        if all(p in _MODULI for p in target_props):
            y_label  = "Composite modulus  (MPa)"
            y_scale  = 1.0
        elif all(p in _CTE for p in target_props):
            y_label  = "CTE  (ppm/K)"
            y_scale  = 1e6
        else:
            y_label  = "Property value"
            y_scale  = 1.0

    # Wider figure when multiple properties share one axis
    fig_h = _H + 0.2 * max(0, len(target_props) - 2)
    fig, ax = plt.subplots(figsize=(_W, fig_h))

    for i, prop in enumerate(target_props):
        color = _COLORS[i % len(_COLORS)]
        ls    = _PROP_LINESTYLES[i % len(_PROP_LINESTYLES)]
        y     = [v * y_scale for v in results[prop]]
        ax.plot(sweep_vals, y, color=color, ls=ls, lw=_LWIDTH,
                marker="o", markersize=_MSIZE - 1, label=prop)

    ax.set_xlabel(_PARAM_AXIS_LABELS.get(parameter, parameter), fontsize=_FONT)
    ax.set_ylabel(y_label, fontsize=_FONT)
    ax.tick_params(labelsize=_FONT - 1)
    ax.legend(fontsize=_FONT - 1, frameon=False,
              loc="best", ncol=min(3, len(target_props)))
    ax.grid(axis="y", lw=0.4, alpha=0.4, color="grey")
    _despine(ax)
    fig.tight_layout(pad=0.6)

    props_label = " / ".join(target_props)
    _open_in_browser(fig, f"{props_label} vs {parameter}")

    # Return numerical table so the agent can describe trends accurately
    # (the browser plot has no return channel — this is the only data the agent sees)
    param_label = _PARAM_AXIS_LABELS.get(parameter, parameter).split("  ")[0]
    col_w = max(10, max(len(p) for p in target_props) + 2)
    header = f"{'':>12}" + "".join(f"{p:>{col_w}}" for p in target_props)
    sep    = "-" * len(header)
    rows   = [header, sep]
    for i, x in enumerate(sweep_vals):
        row = f"{x:>12.4g}"
        for p in target_props:
            v = results[p][i] * y_scale
            row += f"{v:>{col_w}.4g}"
        rows.append(row)

    return "\n".join(rows)
