"""
gui_thermal_inverse.py
======================
Popup window for temperature-dependent thermal inverse estimation.

Shows three live plots after estimation:
  1. Composite K11/K22/K33 — surrogate prediction vs. measured data
  2. Fiber conductivities (longitudinal and transverse) vs. temperature
  3. Polymer (matrix) conductivity vs. temperature

Can be launched standalone or opened from gui_inverse.py.

Usage
-----
    python gui_thermal_inverse.py
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import db.db as _db
from core.unit_manager import UM

_HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("JAX_ENABLE_X64",    "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

FONT_TITLE  = ("Helvetica", 16, "bold")
FONT_LABEL  = ("Helvetica", 13)
FONT_BOLD   = ("Helvetica", 13, "bold")
FONT_SMALL  = ("Helvetica", 11)
FONT_ENTRY  = ("Helvetica", 12)
FONT_MONO   = ("Courier",   11)
FONT_STATUS = ("Helvetica", 12, "italic")


# ═════════════════════════════════════════════════════════════════════════════
# ThermalInverseWindow
# ═════════════════════════════════════════════════════════════════════════════

class ThermalInverseWindow:
    """
    Toplevel popup for thermal inverse estimation.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget (pass None to create a standalone root window).
    """

    def __init__(self, parent=None):
        if parent is None:
            self._win = tk.Tk()
            self._win.title("Thermal Inverse Estimation")
        else:
            self._win = tk.Toplevel(parent)
            self._win.title("Thermal Inverse Estimation")
            self._win.transient(parent)

        self._win.geometry("1440x860")
        self._win.minsize(1100, 720)

        # state
        self._fwd       = None   # ForwardModel (loaded lazily)
        self._predictor = None   # batched predictor
        self._result: Optional[dict] = None
        self._fixed_inputs: Optional[dict] = None
        self._card_fiber_id:   Optional[int] = None
        self._card_polymer_id: Optional[int] = None

        # dynamic unit StringVars (updated via UM callback)
        self._k_unit_var:       tk.StringVar | None = None
        self._density_unit_var: tk.StringVar | None = None

        self._build_ui()
        UM.register_callback(self._refresh_unit_labels)

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        win = self._win
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=0, minsize=380)
        win.grid_columnconfigure(1, weight=1)

        # ── left panel ────────────────────────────────────────────────────────
        left = ttk.Frame(win, padding=(10, 8))
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_columnconfigure(1, weight=1)

        row = 0
        tk.Label(left, text="Thermal Inverse Estimation",
                 font=FONT_TITLE).grid(row=row, column=0, columnspan=3,
                                       sticky="w", pady=(0, 4))
        row += 1
        units_f = ttk.Frame(left)
        units_f.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        tk.Label(units_f, text="Units:", font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self._unit_sys_var = tk.StringVar(value=UM.current_system)
        self._unit_sys_cb  = ttk.Combobox(units_f, textvariable=self._unit_sys_var,
                                           values=UM.available_systems,
                                           state="readonly", width=38, font=FONT_SMALL)
        self._unit_sys_cb.pack(side="left")
        self._unit_sys_cb.bind("<<ComboboxSelected>>", self._on_unit_system_change)

        # initialise dynamic unit StringVars
        self._k_unit_var       = tk.StringVar(value=UM.unit_label("k_f1"))
        self._density_unit_var = tk.StringVar(value=UM.unit_label("rho_f"))

        # ── model load ────────────────────────────────────────────────────────
        row += 1
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1
        self._model_status = tk.Label(
            left, text="Thermal model: not loaded", font=FONT_STATUS, fg="gray")
        self._model_status.grid(row=row, column=0, columnspan=2, sticky="w")
        self._load_btn = ttk.Button(left, text="Load Model",
                                    command=self._on_load_model)
        self._load_btn.grid(row=row, column=2, sticky="e", padx=(4, 0))

        # ── data file ─────────────────────────────────────────────────────────
        row += 1
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1
        tk.Label(left, text="Measurement data file:", font=FONT_LABEL).grid(
            row=row, column=0, columnspan=3, sticky="w")
        row += 1
        self._data_var = tk.StringVar()
        tk.Entry(left, textvariable=self._data_var,
                 font=FONT_ENTRY, width=28).grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=(0, 4))
        ttk.Button(left, text="Browse…",
                   command=self._browse_data).grid(row=row, column=2, sticky="e")
        row += 1
        tk.Label(left,
                 text="CSV/Excel — columns: Temperature, K11, K22, K33\n"
                      "(K22 and K33 are optional)",
                 font=FONT_SMALL, fg="gray").grid(
            row=row, column=0, columnspan=3, sticky="w")

        # ── structural parameters ─────────────────────────────────────────────
        row += 1
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        row += 1
        tk.Label(left, text="Fixed structural parameters",
                 font=FONT_BOLD).grid(row=row, column=0, columnspan=2,
                                      sticky="w", pady=(0, 4))
        ttk.Button(left, text="📂 Load from Card",
                   command=self._on_load_from_card).grid(
            row=row, column=2, sticky="e", pady=(0, 4))

        def _add_field(r, label, var_default, unit_var=None, unit_static=""):
            tk.Label(left, text=label + ":", font=FONT_LABEL, anchor="e",
                     width=22).grid(row=r, column=0, sticky="e", pady=2)
            var = tk.StringVar(value=var_default)
            tk.Entry(left, textvariable=var,
                     font=FONT_ENTRY, width=11).grid(
                row=r, column=1, sticky="w", padx=(4, 2), pady=2)
            if unit_var is not None:
                tk.Label(left, textvariable=unit_var,
                         font=FONT_SMALL, fg="gray").grid(row=r, column=2, sticky="w")
            elif unit_static:
                tk.Label(left, text=unit_static,
                         font=FONT_SMALL, fg="gray").grid(row=r, column=2, sticky="w")
            return var

        row += 1; self._ar_var   = _add_field(row, "Aspect ratio",       "20.0")
        row += 1; self._vf_var   = _add_field(row, "Volume fraction vf", "0.174")
        row += 1; self._rhof_var = _add_field(row, "Fiber density",  "1780",
                                               unit_var=self._density_unit_var)
        row += 1; self._rhom_var = _add_field(row, "Matrix density", "1280",
                                               unit_var=self._density_unit_var)

        row += 1
        tk.Label(left, text="Orientation tensor:", font=FONT_BOLD).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(6, 2))

        def _ot_row(r, labels_defaults):
            f = ttk.Frame(left)
            f.grid(row=r, column=0, columnspan=3, sticky="w", padx=4)
            vars_ = []
            for lbl, default in labels_defaults:
                tk.Label(f, text=lbl + ":", font=FONT_SMALL,
                         width=4, anchor="e").pack(side="left")
                v = tk.StringVar(value=default)
                tk.Entry(f, textvariable=v, font=FONT_ENTRY,
                         width=7).pack(side="left", padx=(2, 8))
                vars_.append(v)
            return vars_

        row += 1
        a_vars1 = _ot_row(row, [("a11", "0.333"), ("a22", "0.333")])
        self._a11_var, self._a22_var = a_vars1
        row += 1
        a_vars2 = _ot_row(row, [("a12", "0.0"), ("a13", "0.0"), ("a23", "0.0")])
        self._a12_var, self._a13_var, self._a23_var = a_vars2

        # ── optimisation settings ─────────────────────────────────────────────
        row += 1
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        row += 1
        tk.Label(left, text="Optimisation", font=FONT_BOLD).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))

        row += 1
        tk.Label(left, text="Restarts:", font=FONT_LABEL, anchor="e",
                 width=22).grid(row=row, column=0, sticky="e", pady=2)
        self._restarts_var = tk.StringVar(value="10")
        tk.Entry(left, textvariable=self._restarts_var,
                 font=FONT_ENTRY, width=6).grid(
            row=row, column=1, sticky="w", padx=(4, 0), pady=2)

        row += 1
        tk.Label(left, text="Seed:", font=FONT_LABEL, anchor="e",
                 width=22).grid(row=row, column=0, sticky="e", pady=2)
        self._seed_var = tk.StringVar(value="0")
        tk.Entry(left, textvariable=self._seed_var,
                 font=FONT_ENTRY, width=6).grid(
            row=row, column=1, sticky="w", padx=(4, 0), pady=2)

        # ── run button ────────────────────────────────────────────────────────
        row += 1
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        row += 1
        self._run_btn = ttk.Button(left, text="  Run Estimation  ",
                                   command=self._on_run, state="disabled")
        self._run_btn.grid(row=row, column=0, columnspan=3, pady=(2, 4))

        row += 1
        self._status_var = tk.StringVar(value="Load the thermal model to begin.")
        tk.Label(left, textvariable=self._status_var,
                 font=FONT_STATUS, fg="gray",
                 wraplength=360, justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))

        # ── results display ───────────────────────────────────────────────────
        row += 1
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=4)
        row += 1
        tk.Label(left, text="Estimated parameters", font=FONT_BOLD).grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))

        def _res_row(r, label, unit_var=None, unit_static=""):
            tk.Label(left, text=label + ":", font=FONT_LABEL, anchor="e",
                     width=28).grid(row=r, column=0, sticky="e", pady=1)
            var = tk.StringVar(value="—")
            tk.Label(left, textvariable=var,
                     font=FONT_MONO, width=12, anchor="w").grid(
                row=r, column=1, sticky="w", padx=(4, 2))
            if unit_var is not None:
                tk.Label(left, textvariable=unit_var,
                         font=FONT_SMALL, fg="gray").grid(row=r, column=2, sticky="w")
            elif unit_static:
                tk.Label(left, text=unit_static,
                         font=FONT_SMALL, fg="gray").grid(row=r, column=2, sticky="w")
            return var

        row += 1; self._r_p1  = _res_row(row, "p1  (polymer scaling)", unit_var=self._k_unit_var)
        row += 1; self._r_p2  = _res_row(row, "p2  (polymer offset)",  unit_var=self._k_unit_var)
        row += 1; self._r_l2  = _res_row(row, "l2  (fiber long. K)",   unit_var=self._k_unit_var)
        row += 1; self._r_t   = _res_row(row, "t   (anisotropy ratio)")
        row += 1; self._r_kft = _res_row(row, "K_f_trans  (= l2/t)",   unit_var=self._k_unit_var)
        row += 1; self._r_mse = _res_row(row, "Final MSE",             unit_static="(model units)²")

        # ── save buttons ──────────────────────────────────────────────────────
        row += 1
        ttk.Separator(left, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        row += 1
        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky="w")
        self._save_csv_btn = ttk.Button(btn_frame, text="Save CSV",
                                        command=self._on_save_csv,
                                        state="disabled")
        self._save_csv_btn.pack(side="left", padx=(0, 6))
        self._save_plots_btn = ttk.Button(btn_frame, text="Save Plots",
                                          command=self._on_save_plots,
                                          state="disabled")
        self._save_plots_btn.pack(side="left")
        self._save_card_btn = ttk.Button(btn_frame, text="Save to Card",
                                         command=self._on_save_to_card,
                                         state="disabled")
        self._save_card_btn.pack(side="left", padx=(6, 0))

        # ── right panel: plots ────────────────────────────────────────────────
        right = ttk.Frame(win, padding=(6, 8))
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(8, 7), tight_layout=True)
        self._ax_composite = self._fig.add_subplot(3, 1, 1)
        self._ax_fiber     = self._fig.add_subplot(3, 1, 2)
        self._ax_polymer   = self._fig.add_subplot(3, 1, 3)
        self._draw_placeholder()

        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas.draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Placeholder plots
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        for ax, title in [
            (self._ax_composite, "Composite K11 / K22 / K33  vs. temperature"),
            (self._ax_fiber,     "Fiber conductivities  vs. temperature"),
            (self._ax_polymer,   "Polymer conductivity  vs. temperature"),
        ]:
            ax.cla()
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5, "Run estimation to see results",
                    ha="center", va="center", fontsize=10,
                    color="gray", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _on_load_model(self):
        self._load_btn.config(state="disabled")
        self._status_var.set("Loading thermal model…")
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            from core.forward import load_forward
            from core.inverse_thermal import make_batched_predictor
            fwd       = load_forward("thermal")
            predictor = make_batched_predictor(fwd)
            self._win.after(0, lambda: self._on_load_ok(fwd, predictor))
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self._win.after(0, lambda m=str(exc), t=tb: self._on_load_err(m, t))

    def _on_load_ok(self, fwd, predictor):
        self._fwd       = fwd
        self._predictor = predictor
        self._model_status.config(
            text="Thermal model: loaded  (12 inputs / 6 outputs)", fg="green")
        self._load_btn.config(state="normal")
        self._run_btn.config(state="normal")
        self._status_var.set("Model loaded. Set inputs and click Run.")

    def _on_load_err(self, msg, tb):
        self._model_status.config(text="Thermal model: load failed", fg="red")
        self._load_btn.config(state="normal")
        self._status_var.set("Model load failed.")
        messagebox.showerror("Load Error", f"{msg}\n\n{tb}", parent=self._win)

    # ─────────────────────────────────────────────────────────────────────────
    # Data file browser
    # ─────────────────────────────────────────────────────────────────────────

    def _browse_data(self):
        path = filedialog.askopenfilename(
            parent=self._win,
            title="Select measurement data file",
            filetypes=[("CSV / Excel", "*.csv *.xlsx *.xls"), ("All", "*.*")],
        )
        if path:
            self._data_var.set(path)

    # ─────────────────────────────────────────────────────────────────────────
    # Collect inputs
    # ─────────────────────────────────────────────────────────────────────────

    def _collect(self):
        def _f(var, name):
            try:
                return float(var.get())
            except ValueError:
                raise ValueError(f"Invalid value for '{name}'.")

        vf    = _f(self._vf_var,   "volume fraction vf")
        rho_f = UM.from_display("rho_f", _f(self._rhof_var, "fiber density"))
        rho_m = UM.from_display("rho_m", _f(self._rhom_var, "matrix density"))

        from core.inverse_thermal import vf_to_wf
        w_f = vf_to_wf(vf, rho_f, rho_m)

        fixed_inputs = {
            "ar_f":  _f(self._ar_var,  "aspect ratio"),
            "w_f":   w_f,
            "rho_f": rho_f,
            "rho_m": rho_m,
            "a11":   _f(self._a11_var, "a11"),
            "a22":   _f(self._a22_var, "a22"),
            "a12":   _f(self._a12_var, "a12"),
            "a13":   _f(self._a13_var, "a13"),
            "a23":   _f(self._a23_var, "a23"),
        }

        n_restarts = int(float(self._restarts_var.get()))
        seed       = int(float(self._seed_var.get()))
        data_path  = self._data_var.get().strip()

        if not data_path:
            raise ValueError("No measurement data file selected.")
        if not os.path.isfile(data_path):
            raise ValueError(f"Data file not found:\n{data_path}")

        return fixed_inputs, n_restarts, seed, data_path, vf

    # ─────────────────────────────────────────────────────────────────────────
    # Run
    # ─────────────────────────────────────────────────────────────────────────

    def _on_run(self):
        if self._predictor is None:
            messagebox.showwarning("No model", "Load the thermal model first.",
                                   parent=self._win)
            return
        try:
            fixed_inputs, n_restarts, seed, data_path, vf = self._collect()
        except ValueError as exc:
            messagebox.showerror("Input error", str(exc), parent=self._win)
            return

        self._fixed_inputs = fixed_inputs
        self._run_btn.config(state="disabled")
        self._save_csv_btn.config(state="disabled")
        self._save_plots_btn.config(state="disabled")
        self._status_var.set("Running… (restart 0)")
        self._reset_results()

        threading.Thread(
            target=self._run_worker,
            args=(fixed_inputs, n_restarts, seed, data_path),
            daemon=True,
        ).start()

    def _run_worker(self, fixed_inputs, n_restarts, seed, data_path):
        try:
            from core.inverse_thermal import (
                make_batched_predictor,
                compute_composite_conductivity,
                run_inverse_estimation,
                PolymerConductivityModel,
                FiberConductivityModel,
            )
            import pandas as pd

            # load data
            ext = os.path.splitext(data_path)[1].lower()
            df  = pd.read_excel(data_path) if ext in (".xlsx", ".xls") \
                  else pd.read_csv(data_path)
            df.columns = [c.strip().lower() for c in df.columns]
            temperatures = df["temperature"].to_numpy(dtype=float)
            K_data = {
                k: df[k.lower()].to_numpy(dtype=float)
                   if k.lower() in df.columns else None
                for k in ("K11", "K22", "K33")
            }

            def _progress(i, n, loss):
                self._win.after(
                    0, lambda i=i, n=n, l=loss:
                        self._status_var.set(
                            f"Restart {i}/{n} — loss = {l:.4e}"))

            best_params, best_loss = run_inverse_estimation(
                temperatures  = temperatures,
                K_data        = K_data,
                predictor     = self._predictor,
                fixed_inputs  = fixed_inputs,
                n_restarts    = n_restarts,
                seed          = seed,
                progress_cb   = _progress,
            )

            K_pred = compute_composite_conductivity(
                best_params, temperatures, self._predictor, fixed_inputs)

            self._win.after(0, lambda: self._on_run_ok(
                best_params, best_loss, temperatures, K_data, K_pred))

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self._win.after(0, lambda m=str(exc), t=tb: self._on_run_err(m, t))

    def _on_run_ok(self, best_params, best_loss,
                   temperatures, K_data, K_pred):
        from core.inverse_thermal import PolymerConductivityModel, FiberConductivityModel

        # store for save functions
        self._result = dict(
            best_params  = best_params,
            best_loss    = best_loss,
            temperatures = temperatures,
            K_data       = K_data,
            K_pred       = K_pred,
        )

        # update result labels (convert conductivities to display units)
        self._r_p1.set( f"{UM.to_display('k_m',  best_params.p1):.6g}")
        self._r_p2.set( f"{UM.to_display('k_m',  best_params.p2):.6g}")
        self._r_l2.set( f"{UM.to_display('k_f1', best_params.l2):.6g}")
        self._r_t.set(  f"{best_params.t:.6g}")
        self._r_kft.set(f"{UM.to_display('k_f1', best_params.l2 / best_params.t):.6g}")
        self._r_mse.set(f"{best_loss:.4e}")

        # draw plots
        self._draw_results(best_params, temperatures, K_data, K_pred)

        self._status_var.set(f"Done. MSE = {best_loss:.4e}")
        self._run_btn.config(state="normal")
        self._save_csv_btn.config(state="normal")
        self._save_plots_btn.config(state="normal")
        self._save_card_btn.config(state="normal")

    def _on_run_err(self, msg, tb):
        self._status_var.set("Estimation failed.")
        self._run_btn.config(state="normal")
        messagebox.showerror("Run Error", f"{msg}\n\n{tb}", parent=self._win)

    # ─────────────────────────────────────────────────────────────────────────
    # Plotting
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_results(self, best_params, temperatures, K_data, K_pred):
        from core.inverse_thermal import PolymerConductivityModel, FiberConductivityModel

        poly    = PolymerConductivityModel(best_params.p1, best_params.p2)
        fiber   = FiberConductivityModel(best_params.l2, best_params.t)
        T_lo    = temperatures.min()
        T_hi    = temperatures.max()
        T_pad   = max((T_hi - T_lo) * 0.05, 5.0)
        T_dense = np.linspace(T_lo - T_pad, T_hi + T_pad, 300)

        k_unit  = UM.unit_label("k11")
        k_fac   = UM.get_factor("k11")
        k_ylabel = f"K  [{k_unit}]"

        # ── subplot 1: composite conductivities ───────────────────────────────
        ax = self._ax_composite
        ax.cla()
        colours = {"K11": "#2176AE", "K22": "#E87040", "K33": "#3EA055"}
        labels  = {"K11": r"$K_{11}$", "K22": r"$K_{22}$", "K33": r"$K_{33}$"}
        for key, col in [("K11", 0), ("K22", 1), ("K33", 2)]:
            c  = colours[key]
            lb = labels[key]
            ax.plot(temperatures, K_pred[:, col] * k_fac, color=c, lw=2.2,
                    label=f"{lb} (surrogate)")
            if K_data.get(key) is not None:
                ax.scatter(temperatures, K_data[key] * k_fac, color=c,
                           s=60, zorder=5, marker="o",
                           label=f"{lb} (measured)")
        ax.set_xlabel("Temperature  (°C)", fontsize=9)
        ax.set_ylabel(k_ylabel, fontsize=9)
        ax.set_title("Composite thermal conductivities: surrogate vs. measured data",
                     fontsize=10)
        ax.legend(fontsize=8, ncol=3, loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # ── subplot 2: fiber conductivities ───────────────────────────────────
        ax = self._ax_fiber
        ax.cla()
        kfl = fiber.K_f_long(T_dense)  * k_fac
        kft = fiber.K_f_trans(T_dense) * k_fac
        l2d = UM.to_display("k_f1", best_params.l2)
        kftd = UM.to_display("k_f1", best_params.l2 / best_params.t)
        ax.plot(T_dense, kfl, lw=2.5, color="#7B2D8B", ls="-",
                label=rf"$K_{{f,\,\mathrm{{long}}}}$ = {l2d:.4g} {k_unit}  [constant]")
        ax.plot(T_dense, kft, lw=2.5, color="#7B2D8B", ls="--",
                label=rf"$K_{{f,\,\mathrm{{trans}}}}$ = {kftd:.4g} {k_unit}  [constant]")
        ax.text(0.97, 0.55,
                f"l\u2082 = {l2d:.4g} {k_unit}\n"
                f"t  = {best_params.t:.4g}  (anisotropy ratio)\n"
                f"K_f_trans = l\u2082/t = {kftd:.4g} {k_unit}",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEE8F8", alpha=0.9))
        ax.set_xlabel("Temperature  (°C)", fontsize=9)
        ax.set_ylabel(k_ylabel, fontsize=9)
        ax.set_title("Fiber thermal conductivity vs. temperature  "
                     "(temperature-independent)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # ── subplot 3: polymer conductivity ───────────────────────────────────
        ax = self._ax_polymer
        ax.cla()
        km = poly(T_dense) * k_fac
        p1d = UM.to_display("k_m", best_params.p1)
        p2d = UM.to_display("k_m", best_params.p2)
        ax.plot(T_dense, km, lw=2.5, color="#C0392B",
                label=r"$K_m(T) = p_1\,\sqrt{T / T_\mathrm{ref}} + p_2$")
        T_meas_range = np.linspace(T_lo, T_hi, 60)
        ax.fill_between(T_meas_range, poly(T_meas_range) * k_fac,
                        alpha=0.12, color="#C0392B", label="measured T range")
        ax.text(0.97, 0.12,
                f"p\u2081 = {p1d:.4g} {k_unit}\n"
                f"p\u2082 = {p2d:.4g} {k_unit}\n"
                f"T_ref = 1.0 \u00b0C",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FDECEA", alpha=0.9))
        ax.set_xlabel("Temperature  (°C)", fontsize=9)
        ax.set_ylabel(k_ylabel, fontsize=9)
        ax.set_title("Polymer (matrix) thermal conductivity vs. temperature  "
                     r"[$K_m(T) = p_1\sqrt{T} + p_2$]", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        self._fig.tight_layout(pad=1.4)
        self._canvas.draw()

    def _reset_results(self):
        for var in (self._r_p1, self._r_p2, self._r_l2,
                    self._r_t, self._r_kft, self._r_mse):
            var.set("—")
        self._draw_placeholder()
        self._canvas.draw()
        self._save_card_btn.config(state="disabled")

    # ─────────────────────────────────────────────────────────────────────────
    # Unit system
    # ─────────────────────────────────────────────────────────────────────────

    def _on_unit_system_change(self, _event=None):
        new_sys = self._unit_sys_var.get()
        old_sys = UM.current_system
        if new_sys == old_sys:
            return
        # Convert density fields
        for var, field in [(self._rhof_var, "rho_f"), (self._rhom_var, "rho_m")]:
            raw = var.get().strip()
            try:
                new_val = UM.convert_between(field, float(raw), old_sys, new_sys)
                var.set(f"{new_val:.6g}")
            except ValueError:
                pass
        UM.set_system(new_sys)   # fires callbacks → _refresh_unit_labels
        # Redraw plots with new units if results exist
        if self._result is not None:
            r = self._result
            self._draw_results(r["best_params"], r["temperatures"],
                               r["K_data"], r["K_pred"])

    def _refresh_unit_labels(self):
        """Called by UM when any window changes the unit system."""
        if hasattr(self, "_unit_sys_var"):
            self._unit_sys_var.set(UM.current_system)
        if self._k_unit_var is not None:
            self._k_unit_var.set(UM.unit_label("k_f1"))
        if self._density_unit_var is not None:
            self._density_unit_var.set(UM.unit_label("rho_f"))
        # Re-display numeric results with new units
        self._redisplay_results()

    def _redisplay_results(self):
        """Re-render the result labels using the current unit system."""
        if self._result is None:
            return
        bp = self._result["best_params"]
        self._r_p1.set( f"{UM.to_display('k_m',  bp.p1):.6g}")
        self._r_p2.set( f"{UM.to_display('k_m',  bp.p2):.6g}")
        self._r_l2.set( f"{UM.to_display('k_f1', bp.l2):.6g}")
        self._r_t.set(  f"{bp.t:.6g}")
        self._r_kft.set(f"{UM.to_display('k_f1', bp.l2 / bp.t):.6g}")
        self._r_mse.set(f"{self._result['best_loss']:.4e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────

    def _on_save_csv(self):
        if self._result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self._win,
            title="Save CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="thermal_inverse_results.csv",
        )
        if not path:
            return
        try:
            import pandas as pd
            from core.inverse_thermal import PolymerConductivityModel, FiberConductivityModel
            r    = self._result
            bp   = r["best_params"]
            T    = r["temperatures"]
            poly = PolymerConductivityModel(bp.p1, bp.p2)
            fib  = FiberConductivityModel(bp.l2, bp.t)
            nan_ = np.full(len(T), np.nan)
            df   = pd.DataFrame({
                "Temperature":   T,
                "K_polymer":     poly(T),
                "K_fiber_long":  fib.K_f_long(T),
                "K_fiber_trans": fib.K_f_trans(T),
                "K11_pred":      r["K_pred"][:, 0],
                "K22_pred":      r["K_pred"][:, 1],
                "K33_pred":      r["K_pred"][:, 2],
                "K11_data":      r["K_data"]["K11"] if r["K_data"]["K11"] is not None else nan_,
                "K22_data":      r["K_data"]["K22"] if r["K_data"]["K22"] is not None else nan_,
                "K33_data":      r["K_data"]["K33"] if r["K_data"]["K33"] is not None else nan_,
            })
            df.to_csv(path, index=False, float_format="%.6f")
            self._status_var.set(f"CSV saved: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc), parent=self._win)

    def _on_save_plots(self):
        if self._result is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self._win,
            title="Save figure",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")],
            initialfile="thermal_inverse_results.png",
        )
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=220, bbox_inches="tight")
            self._status_var.set(f"Figure saved: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc), parent=self._win)

    def _on_load_from_card(self):
        if not _db.db_exists():
            messagebox.showwarning("No database",
                                   "Run  python init_db.py  first.",
                                   parent=self._win)
            return
        from gui_card_dialogs import LoadFromCardDialog
        dlg = LoadFromCardDialog(self._win)
        if not dlg.loaded:
            return

        loaded = dlg.loaded

        # Aspect ratio
        for key in ("ar_f", "ar"):
            if key in loaded:
                self._ar_var.set(f"{loaded[key]:.6g}")
                break

        # Densities (convert from SI model units to current display units).
        # Card may store either the thermal model name (rho_f/rho_m) or the
        # constituent library name (fiber_density/matrix_density).
        for key in ("rho_f", "fiber_density"):
            if key in loaded:
                self._rhof_var.set(f"{UM.to_display('rho_f', loaded[key]):.6g}")
                break
        for key in ("rho_m", "matrix_density"):
            if key in loaded:
                self._rhom_var.set(f"{UM.to_display('rho_m', loaded[key]):.6g}")
                break

        # Volume fraction: card stores w_f (mass fraction); convert back to vf.
        # Must use the densities we just populated, converted back to SI.
        for key in ("fiber_massfrac", "w_f"):
            if key in loaded:
                wf = loaded[key]
                try:
                    rho_f = UM.from_display("rho_f", float(self._rhof_var.get()))
                    rho_m = UM.from_display("rho_m", float(self._rhom_var.get()))
                    # inverse of vf_to_wf: vf = wf*rho_m / (rho_f*(1-wf) + rho_m*wf)
                    vf = wf * rho_m / (rho_f * (1.0 - wf) + rho_m * wf)
                    self._vf_var.set(f"{vf:.6g}")
                except (ValueError, ZeroDivisionError):
                    pass
                break

        # Orientation tensor
        for field, var in [
            ("a11", self._a11_var), ("a22", self._a22_var),
            ("a12", self._a12_var), ("a13", self._a13_var), ("a23", self._a23_var),
        ]:
            if field in loaded:
                var.set(f"{loaded[field]:.6g}")

        # Remember which card this came from (for save-to-card)
        if dlg.loaded_card is not None:
            self._card_fiber_id   = dlg.loaded_card.get("fiber_id")
            self._card_polymer_id = dlg.loaded_card.get("polymer_id")

    def _on_save_to_card(self):
        if self._result is None:
            return
        if not _db.db_exists():
            messagebox.showwarning("No database",
                                   "Run  python init_db.py  first.",
                                   parent=self._win)
            return
        from gui_card_dialogs import SaveThermalToCardDialog
        r = self._result
        SaveThermalToCardDialog(
            self._win,
            params       = r["best_params"],
            fixed_inputs = self._fixed_inputs or {},
            temperatures = r["temperatures"],
            K_pred       = r["K_pred"],
            loss         = r["best_loss"],
            fiber_id     = self._card_fiber_id,
            polymer_id   = self._card_polymer_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────────

    def mainloop(self):
        self._win.mainloop()

    def wait_window(self):
        self._win.wait_window()


# ═════════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    root.withdraw()   # hide bare root; the Toplevel is the real window
    win = ThermalInverseWindow(parent=None)
    win._win.lift()
    win._win.attributes("-topmost", True)
    win._win.after(200, lambda: win._win.attributes("-topmost", False))
    win._win.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
