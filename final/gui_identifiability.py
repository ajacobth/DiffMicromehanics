"""gui_identifiability.py – Identifiability analysis window.

Open via gui.py "Identifiability Check" button.  Requires a loaded ForwardModel.
"""
from __future__ import annotations

import json
import os
import sys
import threading

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── constants ─────────────────────────────────────────────────────────────────

# Scale factors: model-native (MPa, 1/K) -> display units
OUTPUT_SCALES: dict[str, float] = {
    "E1":  1e-3, "E2":  1e-3, "E3":  1e-3,
    "G12": 1e-3, "G13": 1e-3, "G23": 1e-3,
    "nu12": 1.0, "nu13": 1.0, "nu23": 1.0,
    "CTE11": 1e6, "CTE22": 1e6, "CTE33": 1e6,
    "CTE12": 1e6, "CTE13": 1e6, "CTE23": 1e6,
}

COLOR_WELL     = "#1f7a1f"
COLOR_MARGINAL = "#b36200"
COLOR_POOR     = "#b30000"
COLOR_BG_WELL  = "#e6f4e6"
COLOR_BG_MARG  = "#fff3e0"
COLOR_BG_POOR  = "#fce4e4"

FONT_TITLE  = ("Helvetica", 14, "bold")
FONT_LABEL  = ("Helvetica", 12)
FONT_BOLD   = ("Helvetica", 12, "bold")
FONT_STATUS = ("Helvetica", 11, "italic")
FONT_ENTRY  = ("Helvetica", 12)
FONT_SMALL  = ("Helvetica", 10)
FONT_MONO   = ("Courier", 11)
FONT_VERDICT = ("Helvetica", 13, "bold")


def _load_field_labels():
    path = os.path.join(_HERE, "field_labels.json")
    try:
        with open(path) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _load_problem():
    path = os.path.join(_HERE, "problem.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _label(labels, section, key):
    return labels.get(section, {}).get(key, key)


def _output_unit_label(name, labels):
    """Extract the unit part from a label, e.g. 'E1 (GPa)' -> 'GPa'."""
    full = _label(labels, "outputs", name)
    if "(" in full and ")" in full:
        return full.split("(")[-1].rstrip(")")
    return ""


# ── main window class ─────────────────────────────────────────────────────────

class IdentifiabilityWindow:
    """Standalone Identifiability Analysis window."""

    def __init__(self, parent, model=None):
        self.parent = parent
        self.model  = model

        self._labels  = _load_field_labels()
        self._problem = _load_problem()
        self._result  = None

        # per-widget state
        self._free_vars:   dict[str, tk.BooleanVar] = {}
        self._target_vars: dict[str, tk.BooleanVar] = {}
        self._sigma_vars:  dict[str, tk.StringVar]  = {}
        self._n_samples_var = tk.StringVar(value="200")
        self._advanced_var  = tk.BooleanVar(value=False)

        self.win = tk.Toplevel(parent)
        self.win.title("Identifiability Check")
        self.win.geometry("1200x900")
        self.win.minsize(900, 650)
        self.win.grid_rowconfigure(1, weight=1)
        self.win.grid_columnconfigure(0, weight=1)

        self._build_ui()

        if model is None:
            messagebox.showwarning(
                "No Model",
                "No model loaded.\nLoad a model in the main window first, "
                "then open the Identifiability Check.",
                parent=self.win,
            )

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        win = self.win

        # ── title bar ──
        title_frame = ttk.Frame(win, padding=(12, 8))
        title_frame.grid(row=0, column=0, sticky="ew")
        tk.Label(
            title_frame,
            text="Identifiability Check",
            font=FONT_TITLE,
        ).pack(side="left")
        tk.Label(
            title_frame,
            text="Plan which experiments can identify which parameters.",
            font=FONT_STATUS, fg="gray",
        ).pack(side="left", padx=16)

        ttk.Separator(win, orient="horizontal").grid(row=0, column=0, sticky="ew", pady=(48, 0))

        # ── scrollable main area ───────────────────────────────────────────
        main_canvas = tk.Canvas(win, borderwidth=0, highlightthickness=0)
        main_vsb    = ttk.Scrollbar(win, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=main_vsb.set)
        main_vsb.grid(row=1, column=1, sticky="ns")
        main_canvas.grid(row=1, column=0, sticky="nsew")

        self._main_frame = ttk.Frame(main_canvas)
        self._main_canvas_win = main_canvas.create_window(
            (0, 0), window=self._main_frame, anchor="nw"
        )
        self._main_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        main_canvas.bind(
            "<Configure>",
            lambda e: main_canvas.itemconfig(self._main_canvas_win, width=e.width)
        )
        main_canvas.bind(
            "<MouseWheel>",
            lambda e: main_canvas.yview_scroll(int(-e.delta / 120), "units")
        )

        self._main_canvas = main_canvas
        self._build_selection_panels(self._main_frame)
        self._build_controls(self._main_frame)
        self._build_results_area(self._main_frame)
        self._build_advanced_area(self._main_frame)

    def _build_selection_panels(self, parent):
        panels = ttk.Frame(parent)
        panels.grid(row=0, column=0, sticky="ew", padx=10, pady=(12, 4))
        panels.grid_columnconfigure(0, weight=1, minsize=350)
        panels.grid_columnconfigure(1, weight=0)
        panels.grid_columnconfigure(2, weight=1, minsize=350)

        # ── left: free variables ───────────────────────────────────────────
        left = ttk.LabelFrame(panels, text="Free Variables (what to infer)", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        lf = ttk.Frame(left)
        lf.grid(row=0, column=0, sticky="nsew")

        if self.model is not None:
            for i, name in enumerate(self.model.input_fields):
                var = tk.BooleanVar(value=False)
                self._free_vars[name] = var
                display = _label(self._labels, "inputs", name)
                cb = ttk.Checkbutton(lf, text=display, variable=var)
                cb.grid(row=i, column=0, sticky="w", padx=4, pady=2)
        else:
            tk.Label(lf, text="(load a model first)", font=FONT_STATUS, fg="gray").pack()

        # ── separator ──
        ttk.Separator(panels, orient="vertical").grid(row=0, column=1, sticky="ns", padx=6)

        # ── right: target outputs ──────────────────────────────────────────
        right = ttk.LabelFrame(panels, text="Target Outputs (what you will measure)", padding=8)
        right.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        right.grid_columnconfigure(0, weight=1)

        rf = ttk.Frame(right)
        rf.grid(row=0, column=0, sticky="nsew")
        rf.grid_columnconfigure(0, weight=1)

        if self.model is not None:
            for i, name in enumerate(self.model.output_fields):
                var = tk.BooleanVar(value=False)
                self._target_vars[name] = var

                display  = _label(self._labels, "outputs", name)
                unit_str = _output_unit_label(name, self._labels)

                row_f = ttk.Frame(rf)
                row_f.grid(row=i, column=0, sticky="ew", pady=2)
                row_f.grid_columnconfigure(0, weight=1)

                cb = ttk.Checkbutton(row_f, text=display, variable=var)
                cb.grid(row=0, column=0, sticky="w")

                # sigma entry (starts blank; user fills in or uses "Help me get started")
                sigma_var = tk.StringVar()
                self._sigma_vars[name] = sigma_var

                sig_frame = ttk.Frame(row_f)
                sig_frame.grid(row=0, column=1, sticky="e", padx=(8, 0))

                tk.Label(sig_frame, text="σ:", font=FONT_SMALL).pack(side="left")
                sig_ent = tk.Entry(sig_frame, textvariable=sigma_var,
                                   width=8, font=FONT_ENTRY)
                sig_ent.pack(side="left", padx=2)
                if unit_str:
                    tk.Label(sig_frame, text=unit_str, font=FONT_SMALL, fg="gray").pack(side="left")
        else:
            tk.Label(rf, text="(load a model first)", font=FONT_STATUS, fg="gray").pack()

    def _build_controls(self, parent):
        ctrl = ttk.Frame(parent, padding=(10, 6))
        ctrl.grid(row=1, column=0, sticky="ew")

        ttk.Button(ctrl, text="Help me get started",
                   command=self._on_help).pack(side="left", padx=(0, 20))

        tk.Label(ctrl, text="N samples:", font=FONT_LABEL).pack(side="left")
        tk.Entry(ctrl, textvariable=self._n_samples_var, width=6,
                 font=FONT_ENTRY).pack(side="left", padx=(4, 20))

        ttk.Checkbutton(ctrl, text="Advanced plots",
                        variable=self._advanced_var).pack(side="left", padx=(0, 20))

        self._run_btn = ttk.Button(ctrl, text="Run Check",
                                   command=self._on_run, state="normal")
        self._run_btn.pack(side="left", padx=(0, 12))

        self._run_status = tk.Label(ctrl, text="", font=FONT_STATUS, fg="orange")
        self._run_status.pack(side="left")

    def _on_help(self):
        answer = messagebox.askyesno(
            "Help me get started",
            "Load predefined fields as an example?",
            parent=self.win,
        )
        if not answer:
            return

        # ── pre-check free variables from problem.json ─────────────────────
        free_in_problem = self._problem.get("free_inputs", [])
        for name, var in self._free_vars.items():
            var.set(name in free_in_problem)

        # ── pre-check target outputs + populate sigmas ─────────────────────
        target_in_problem = self._problem.get("target_outputs", {})
        prob_sigmas = {
            k: v for k, v in self._problem.get("sigmas", {}).items()
            if not isinstance(v, str)
        }
        for name, var in self._target_vars.items():
            var.set(name in target_in_problem)
            if name in target_in_problem:
                model_sigma = prob_sigmas.get(name, 0.0)
                if isinstance(model_sigma, (int, float)) and model_sigma > 0:
                    scale = OUTPUT_SCALES.get(name, 1.0)
                    self._sigma_vars[name].set(f"{model_sigma * scale:.4g}")

    def _build_results_area(self, parent):
        self._results_outer = ttk.LabelFrame(parent, text="Results", padding=10)
        self._results_outer.grid(row=2, column=0, sticky="ew", padx=10, pady=(6, 4))
        self._results_outer.grid_columnconfigure(0, weight=1)

        # placeholder
        self._results_placeholder = tk.Label(
            self._results_outer,
            text="Click 'Run Check' to analyse identifiability.",
            font=FONT_STATUS, fg="gray",
        )
        self._results_placeholder.grid(row=0, column=0, pady=10, sticky="w")

        # verdict (hidden until results available)
        self._verdict_label = tk.Label(
            self._results_outer, text="", font=FONT_VERDICT, wraplength=900, justify="left"
        )

        # treeview for per-parameter table
        cols = ("Parameter", "Status", "CR Uncertainty", "Note")
        self._tree = ttk.Treeview(self._results_outer, columns=cols, show="headings", height=8)
        for c in cols:
            self._tree.heading(c, text=c)
        self._tree.column("Parameter",    width=180, anchor="w")
        self._tree.column("Status",       width=120, anchor="center")
        self._tree.column("CR Uncertainty", width=160, anchor="center")
        self._tree.column("Note",         width=260, anchor="w")

        self._tree.tag_configure("WELL",     background=COLOR_BG_WELL,  foreground=COLOR_WELL)
        self._tree.tag_configure("MARGINAL", background=COLOR_BG_MARG, foreground=COLOR_MARGINAL)
        self._tree.tag_configure("POOR",     background=COLOR_BG_POOR,  foreground=COLOR_POOR)

        # recommendations text
        self._rec_text = tk.Text(
            self._results_outer, font=FONT_MONO, height=8,
            state="disabled", wrap="word", relief="flat", bg="#f9f9f9",
        )

    def _build_advanced_area(self, parent):
        self._adv_outer = ttk.LabelFrame(
            parent, text="Advanced Plots", padding=10
        )
        # hidden by default; shown in _show_advanced_plots
        self._adv_canvas_frame = None

    # ── run analysis ──────────────────────────────────────────────────────────

    def _on_run(self):
        if self.model is None:
            messagebox.showerror("No Model",
                                 "Load a model in the main window first.",
                                 parent=self.win)
            return

        # Collect free variables
        free_inputs = [k for k, v in self._free_vars.items() if v.get()]
        if not free_inputs:
            messagebox.showwarning("No Free Variables",
                                   "Select at least one free variable.",
                                   parent=self.win)
            return

        # Check bounds exist for all free vars
        bounds_raw = {k: tuple(v) for k, v in self._problem.get("bounds", {}).items()
                      if not isinstance(k, str) or not k.startswith("_")}
        missing_bounds = [k for k in free_inputs if k not in bounds_raw]
        if missing_bounds:
            messagebox.showerror(
                "Bounds Required",
                f"Bounds required for identifiability analysis.\n"
                f"Missing bounds for: {', '.join(missing_bounds)}\n\n"
                "Add bounds to problem.json.",
                parent=self.win,
            )
            return

        # Collect target outputs
        target_outputs_display = {k: 0.0 for k, v in self._target_vars.items() if v.get()}
        if not target_outputs_display:
            messagebox.showwarning("No Target Outputs",
                                   "Select at least one target output.",
                                   parent=self.win)
            return

        # Collect sigmas (convert display units -> model units)
        sigmas = {}
        for name in target_outputs_display:
            raw = self._sigma_vars[name].get().strip()
            try:
                sigma_display = float(raw) if raw else 0.0
            except ValueError:
                sigma_display = 0.0
            scale = OUTPUT_SCALES.get(name, 1.0)
            # display = model * scale  =>  model = display / scale
            sigma_model = sigma_display / scale if scale != 0 else sigma_display
            sigmas[name] = max(sigma_model, 0.0)

        # N samples
        try:
            n_samples = int(self._n_samples_var.get())
            n_samples = max(10, min(n_samples, 2000))
        except ValueError:
            n_samples = 200

        show_advanced = self._advanced_var.get()

        # Build x_template from problem.json fixed_inputs
        import jax.numpy as jnp
        fixed = {k: float(v) for k, v in self._problem.get("fixed_inputs", {}).items()
                 if not isinstance(k, str) or not k.startswith("_")}
        x_np = np.zeros(len(self.model.input_fields), dtype=np.float32)
        for k, v in fixed.items():
            if k in self.model.in_idx:
                x_np[self.model.in_idx[k]] = v
        # Set free vars to midpoint of their bounds
        for k in free_inputs:
            if k in bounds_raw and k in self.model.in_idx:
                lo, hi = bounds_raw[k]
                x_np[self.model.in_idx[k]] = (lo + hi) / 2.0
        x_template = jnp.array(x_np, dtype=jnp.float32)

        free_indices = [self.model.in_idx[k] for k in free_inputs]

        self._run_btn.config(state="disabled")
        self._run_status.config(text="Running… (this may take a few seconds)")
        self.win.update_idletasks()

        threading.Thread(
            target=self._run_worker,
            args=(
                self.model.predict_array,
                x_template,
                free_inputs,
                free_indices,
                target_outputs_display,
                self.model.out_idx,
                sigmas,
                bounds_raw,
                self.model.output_fields,
                self.model.output_std,
                n_samples,
                show_advanced,
            ),
            daemon=True,
        ).start()

    def _run_worker(self, predict_array, x_template, free_inputs, free_indices,
                    target_outputs, out_idx, sigmas, bounds,
                    all_output_names, sig_out, n_samples, show_advanced):
        try:
            from fim import run_identifiability_check
            result = run_identifiability_check(
                predict_array, x_template, free_inputs, free_indices,
                target_outputs, out_idx, sigmas, bounds,
                all_output_names, sig_out, N_samples=n_samples,
            )
            self.win.after(0, lambda: self._show_results(
                result, free_inputs, bounds, show_advanced
            ))
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.win.after(0, lambda: self._on_run_error(str(exc), tb))

    def _on_run_error(self, msg, tb):
        self._run_btn.config(state="normal")
        self._run_status.config(text="")
        messagebox.showerror("Analysis Error", f"{msg}\n\n{tb}", parent=self.win)

    # ── display results ───────────────────────────────────────────────────────

    def _show_results(self, result, free_inputs, bounds, show_advanced):
        self._result = result
        self._run_btn.config(state="normal")
        self._run_status.config(text="Done.")

        # Clear placeholder
        self._results_placeholder.grid_forget()

        # ── Overall verdict ────────────────────────────────────────────────
        status_vals = list(result["status"].values())
        n_poor = sum(1 for s in status_vals if s == "POOR")
        n_marg = sum(1 for s in status_vals if s == "MARGINAL")

        if n_poor == 0 and n_marg == 0:
            verdict_text = "✓  Your experiment set can reliably identify all selected free variables."
            verdict_fg   = COLOR_WELL
        elif n_poor > 0:
            verdict_text = (
                f"✗  {n_poor} free variable(s) CANNOT be reliably identified "
                f"with this experiment set.  See recommendations below."
            )
            verdict_fg = COLOR_POOR
        else:
            verdict_text = (
                f"⚠  {n_marg} free variable(s) are only MARGINALLY identified. "
                "Consider adding more experiments."
            )
            verdict_fg = COLOR_MARGINAL

        self._verdict_label.config(text=verdict_text, fg=verdict_fg)
        self._verdict_label.grid(row=0, column=0, sticky="w", pady=(0, 8))

        # ── Per-parameter status table ─────────────────────────────────────
        self._tree.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        # Clear existing rows
        for iid in self._tree.get_children():
            self._tree.delete(iid)

        for k in free_inputs:
            stat    = result["status"][k]
            cr_phys = result["cramer_rao_std"].get(k)
            display = _label(self._labels, "inputs", k)

            if stat == "POOR":
                cr_str  = "—"
                note    = "Cannot be identified"
                sym     = "✗"
            elif stat == "MARGINAL":
                cr_str  = f"±{cr_phys:.4g}" if cr_phys is not None else "—"
                note    = "Marginally identified"
                sym     = "~"
            else:
                cr_str  = f"±{cr_phys:.4g}" if cr_phys is not None else "—"
                note    = "Reliably identified"
                sym     = "✓"

            self._tree.insert(
                "", "end",
                values=(display, f"{sym} {stat}", cr_str, note),
                tags=(stat,),
            )

        # ── Recommendations ────────────────────────────────────────────────
        recs = result.get("recommendations", [])
        has_issues = n_poor > 0 or n_marg > 0

        self._rec_text.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        self._rec_text.config(state="normal")
        self._rec_text.delete("1.0", "end")

        if has_issues and recs:
            self._rec_text.insert("end", "To improve identifiability, consider adding:\n\n")
            for rank, rec in enumerate(recs, 1):
                name   = rec["name"]
                imp    = rec["improvement_factor"]
                disp   = _label(self._labels, "outputs", name)
                line   = f"  {rank}. {name:8s}  ({disp})\n"
                line  += f"        improves worst direction by {imp:.1f}×"
                if imp < 1.01:
                    line += "  (no improvement)"
                line += "\n\n"
                self._rec_text.insert("end", line)
        elif not has_issues:
            self._rec_text.insert("end",
                "All free variables are well-identified. No additional experiments needed.")
        else:
            self._rec_text.insert("end", "No candidate experiments available.")

        self._rec_text.config(state="disabled")

        # ── Advanced plots ─────────────────────────────────────────────────
        if show_advanced:
            self._show_advanced_plots(result, free_inputs)
        else:
            # Hide advanced area if previously shown
            self._adv_outer.grid_forget()

        # scroll to top
        self._main_canvas.yview_moveto(0.0)

    def _show_advanced_plots(self, result, free_inputs):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except ImportError:
            messagebox.showwarning(
                "Matplotlib Missing",
                "Install matplotlib to see advanced plots:\n  pip install matplotlib",
                parent=self.win,
            )
            return

        # Remove old advanced frame content
        for w in self._adv_outer.winfo_children():
            w.destroy()

        self._adv_outer.grid(row=3, column=0, sticky="ew", padx=10, pady=(4, 10))
        self._adv_outer.grid_columnconfigure(0, weight=1)

        # ── Plot 1: FIM Eigenvalue Spectrum ────────────────────────────────
        eigvals = result["eigenvalues"]
        dominant = result["dominant_params"]
        n = len(eigvals)

        colors = []
        for lam in eigvals:
            if lam < 1.0:
                colors.append("#cc2222")
            elif lam < 100.0:
                colors.append("#cc7700")
            else:
                colors.append("#1f7a1f")

        fig1, ax1 = plt.subplots(figsize=(7, 3.5))
        x_pos = np.arange(n)
        ax1.bar(x_pos, eigvals, color=colors)
        ax1.set_yscale("log")
        ax1.axhline(1.0,   color="red",    linestyle="--", label="Identifiability threshold (λ=1)")
        ax1.axhline(100.0, color="orange", linestyle="--", label="Well-identified threshold (λ=100)")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"v{i+1}\n({dominant[i]})" for i in range(n)], fontsize=9)
        ax1.set_ylabel("Eigenvalue (log scale)")
        ax1.set_title("FIM Eigenvalue Spectrum — bars below red line indicate poor identifiability")
        ax1.legend(fontsize=8)
        fig1.tight_layout()

        canvas1 = FigureCanvasTkAgg(fig1, master=self._adv_outer)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=0, column=0, sticky="ew", pady=(4, 8))
        plt.close(fig1)

        # ── Plot 2: Relative Sensitivity Heatmap ───────────────────────────
        S          = result.get("sensitivity_matrix")
        row_names  = result.get("sensitivity_output_names", [])
        col_names  = result.get("sensitivity_param_names", [])

        if S is not None and len(row_names) > 0 and len(col_names) > 0:
            row_labels = [_label(self._labels, "outputs", n) for n in row_names]
            col_labels = [_label(self._labels, "inputs",  n) for n in col_names]

            fig2, ax2 = plt.subplots(figsize=(max(5, len(col_names) * 1.2),
                                               max(3, len(row_names) * 0.7)))
            im = ax2.imshow(S, cmap="YlOrRd", aspect="auto")
            ax2.set_xticks(range(len(col_names)))
            ax2.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
            ax2.set_yticks(range(len(row_names)))
            ax2.set_yticklabels(row_labels, fontsize=9)
            ax2.set_title("Relative Sensitivity — darker = stronger coupling")
            fig2.colorbar(im, ax=ax2)

            # annotate cells
            for i in range(S.shape[0]):
                for j in range(S.shape[1]):
                    ax2.text(j, i, f"{S[i, j]:.2f}",
                             ha="center", va="center", fontsize=8,
                             color="white" if S[i, j] > S.max() * 0.6 else "black")

            fig2.tight_layout()
            canvas2 = FigureCanvasTkAgg(fig2, master=self._adv_outer)
            canvas2.draw()
            canvas2.get_tk_widget().grid(row=1, column=0, sticky="ew", pady=(0, 8))
            plt.close(fig2)

        # ── Plot 3: Eigenvalue Improvement per Candidate Experiment ────────
        recs = result.get("recommendations", [])
        if recs:
            names_r = [r["name"] for r in recs]
            imps_r  = [r["improvement_factor"] for r in recs]
            colors3 = ["#1f7a1f"] + ["#4a90d9"] * (len(names_r) - 1)

            fig3, ax3 = plt.subplots(figsize=(7, max(2.5, len(names_r) * 0.55)))
            y_pos = np.arange(len(names_r))
            ax3.barh(y_pos, imps_r, color=colors3)
            ax3.set_xscale("log")
            ax3.axvline(1.0, color="gray", linestyle="--", label="No improvement")
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(
                [f"{r['name']} ({_label(self._labels, 'outputs', r['name'])})"
                 for r in recs],
                fontsize=9,
            )
            ax3.set_xlabel("Improvement factor (log scale)")
            ax3.set_title("Recommended Next Experiment — improvement to worst-identified direction")
            ax3.legend(fontsize=8)
            ax3.invert_yaxis()
            fig3.tight_layout()

            canvas3 = FigureCanvasTkAgg(fig3, master=self._adv_outer)
            canvas3.draw()
            canvas3.get_tk_widget().grid(row=2, column=0, sticky="ew", pady=(0, 4))
            plt.close(fig3)


# ── launcher ─────────────────────────────────────────────────────────────────

def open_identifiability_window(parent, model=None):
    """Create and return an IdentifiabilityWindow."""
    return IdentifiabilityWindow(parent, model=model)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # hide the root window; the Toplevel is the real UI
    open_identifiability_window(root, model=None)
    root.mainloop()
