"""gui_inverse.py – interactive GUI for the inverse design problem.

The user selects which input variables are free (to be optimised) vs fixed,
enters target output values, adds an optional tag (e.g. "machine:CAMRI"), and
runs the optimiser. Results are displayed and saved to a JSON file.

Usage
-----
    python gui_inverse.py
"""
from __future__ import annotations

import datetime
import json
import os
import threading
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

import db.db as _db
from core.unit_manager import UM

_HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("JAX_ENABLE_X64",    "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# ── field label mapping ────────────────────────────────────────────────────────
_LABELS_FILE = os.path.join(_HERE, "config", "field_labels.json")

def _load_field_labels() -> dict:
    try:
        with open(_LABELS_FILE) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

_FIELD_LABELS = _load_field_labels()

def _label(section: str, key: str) -> str:
    return _FIELD_LABELS.get(section, {}).get(key, key)


# Unit scaling and labels are now provided dynamically by unit_manager.UM.
# See  data/units.json  to customise or add unit systems.

FONT_TITLE  = ("Helvetica", 17, "bold")
FONT_LABEL  = ("Helvetica", 14)
FONT_BOLD   = ("Helvetica", 14, "bold")
FONT_SMALL  = ("Helvetica", 11)
FONT_STATUS = ("Helvetica", 13, "italic")
FONT_ENTRY  = ("Helvetica", 13)
FONT_MONO   = ("Courier",   12)

MODEL_NAMES = ["elastic", "thermoelastic"]
_BIG        = 1e12   # stand-in for ±∞ when only some free vars have explicit bounds


# ── per-input row ──────────────────────────────────────────────────────────────
class InputRow:
    """One row in the Inputs panel (one model input field)."""

    def __init__(self, parent: tk.Widget, row: int, field: str, display: str):
        self.field   = field
        self.display = display
        self._mode   = tk.StringVar(value="fixed")

        # label
        tk.Label(parent, text=f"{display}:", font=FONT_LABEL, anchor="e",
                 width=34).grid(row=row, column=0, sticky="e", padx=(8, 6), pady=2)

        # radio: Fixed | Free
        rb = ttk.Frame(parent)
        rb.grid(row=row, column=1, sticky="w", padx=2)
        ttk.Radiobutton(rb, text="Fixed", variable=self._mode,
                        value="fixed", command=self._on_mode).pack(side="left")
        ttk.Radiobutton(rb, text="Free",  variable=self._mode,
                        value="free",  command=self._on_mode).pack(side="left", padx=(8, 0))

        # value entry + dynamic unit label
        val_f = ttk.Frame(parent)
        val_f.grid(row=row, column=2, padx=6, pady=2)
        self._val_entry = tk.Entry(val_f, width=13, font=FONT_ENTRY)
        self._val_entry.pack(side="left")
        self._unit_lbl = tk.Label(val_f, text=UM.unit_label(field),
                                  font=FONT_SMALL, fg="gray", width=10, anchor="w")
        self._unit_lbl.pack(side="left", padx=(3, 0))

        # bounds frame – hidden until mode = free
        self._bf = ttk.Frame(parent)
        self._bf.grid(row=row, column=3, sticky="w", padx=4)

        tk.Label(self._bf, text="lo:", font=FONT_SMALL).pack(side="left")
        self._lo = tk.Entry(self._bf, width=9, font=FONT_ENTRY)
        self._lo.pack(side="left", padx=(2, 6))

        tk.Label(self._bf, text="hi:", font=FONT_SMALL).pack(side="left")
        self._hi = tk.Entry(self._bf, width=9, font=FONT_ENTRY)
        self._hi.pack(side="left", padx=(2, 0))

        tk.Label(self._bf, text="(optional bounds; leave blank for ±∞)",
                 font=FONT_SMALL, fg="gray").pack(side="left", padx=(10, 0))

        self._bf.grid_remove()

    def _on_mode(self):
        if self._mode.get() == "free":
            self._bf.grid()
        else:
            self._bf.grid_remove()

    def is_free(self) -> bool:
        return self._mode.get() == "free"

    def get_value(self) -> float:
        raw = self._val_entry.get().strip()
        if not raw:
            raise ValueError(f"'{self.display}' has no value.")
        return float(raw)

    def get_bounds(self) -> Optional[tuple]:
        lo = self._lo.get().strip()
        hi = self._hi.get().strip()
        if lo and hi:
            return (float(lo), float(hi))
        return None

    def set_value(self, v: float):
        self._val_entry.delete(0, tk.END)
        self._val_entry.insert(0, f"{v:.6g}")

    def convert_bounds(self, old_sys: str, new_sys: str):
        """Convert bound entries from old_sys to new_sys in-place."""
        lo = self._lo.get().strip()
        hi = self._hi.get().strip()
        if lo:
            try:
                new_lo = UM.convert_between(self.field, float(lo), old_sys, new_sys)
                self._lo.delete(0, tk.END)
                self._lo.insert(0, f"{new_lo:.6g}")
            except ValueError:
                pass
        if hi:
            try:
                new_hi = UM.convert_between(self.field, float(hi), old_sys, new_sys)
                self._hi.delete(0, tk.END)
                self._hi.insert(0, f"{new_hi:.6g}")
            except ValueError:
                pass

    def update_unit_label(self):
        self._unit_lbl.config(text=UM.unit_label(self.field))


# ── per-output row ─────────────────────────────────────────────────────────────
class OutputRow:
    """One row in the Targets panel (one model output field)."""

    def __init__(self, parent: tk.Widget, row: int, field: str, display: str):
        self.field   = field
        self.display = display
        self._active = tk.BooleanVar(value=False)
        self._parent = parent
        self._row    = row

        ttk.Checkbutton(parent, variable=self._active,
                        command=self._on_toggle).grid(row=row, column=0,
                                                      padx=(8, 4), pady=2)
        tk.Label(parent, text=f"{display}:", font=FONT_LABEL, anchor="e",
                 width=34).grid(row=row, column=1, sticky="e", padx=(0, 6), pady=2)

        self._entry = tk.Entry(parent, width=16, font=FONT_ENTRY, state="disabled")
        self._entry.grid(row=row, column=2, padx=6, pady=2)

        # σ (noise) entry — disabled until the row is checked
        sigma_frame = ttk.Frame(parent)
        sigma_frame.grid(row=row, column=3, padx=(4, 2), pady=2, sticky="w")
        tk.Label(sigma_frame, text="σ:", font=FONT_SMALL).pack(side="left")
        self._sigma_entry = tk.Entry(sigma_frame, width=10, font=FONT_ENTRY, state="disabled")
        self._sigma_entry.insert(0, "0")
        self._sigma_entry.pack(side="left", padx=(2, 0))
        unit_str = UM.unit_label(field)
        self._sigma_unit_lbl = tk.Label(sigma_frame, text=unit_str,
                                        font=FONT_SMALL, fg="gray")
        self._sigma_unit_lbl.pack(side="left", padx=(3, 0))

        self._col_unit_lbl = tk.Label(parent,
            text=f"({unit_str})" if unit_str else "(dimensionless)",
            font=FONT_SMALL, fg="gray")
        self._col_unit_lbl.grid(row=row, column=4, sticky="w", padx=2)

    def _on_toggle(self):
        state = "normal" if self._active.get() else "disabled"
        self._entry.config(state=state)
        self._sigma_entry.config(state=state)

    def is_active(self) -> bool:
        return self._active.get()

    def get_target(self) -> float:
        raw = self._entry.get().strip()
        if not raw:
            raise ValueError(f"Target for '{self.display}' is empty.")
        return float(raw)

    def get_sigma(self) -> float:
        raw = self._sigma_entry.get().strip()
        try:
            return float(raw) if raw else 0.0
        except ValueError:
            return 0.0

    def set_target(self, val: float):
        state = str(self._entry["state"])
        self._entry.config(state="normal")
        self._entry.delete(0, tk.END)
        self._entry.insert(0, f"{val:.6g}")
        self._entry.config(state=state)

    def set_sigma(self, val: float):
        state = str(self._sigma_entry["state"])
        self._sigma_entry.config(state="normal")
        self._sigma_entry.delete(0, tk.END)
        self._sigma_entry.insert(0, f"{val:.6g}")
        self._sigma_entry.config(state=state)

    def update_unit_label(self):
        u = UM.unit_label(self.field)
        self._sigma_unit_lbl.config(text=u)
        self._col_unit_lbl.config(text=f"({u})" if u else "(dimensionless)")


# ── main GUI ───────────────────────────────────────────────────────────────────
class InverseGUI:

    def __init__(self, root: tk.Tk):
        self.root         = root
        self.model        = None
        self._input_rows:  list[InputRow]  = []
        self._output_rows: list[OutputRow] = []
        self._last_result: Optional[dict]  = None
        self._card_fiber_id:   Optional[int] = None
        self._card_polymer_id: Optional[int] = None
        self._fiber_map:  dict[str, int] = {}
        self._polymer_map: dict[str, int] = {}

        root.title("Composite Surrogate – Inverse Design")
        root.geometry("1700x1020")
        root.minsize(1200, 700)
        self._build_layout()
        UM.register_callback(self._refresh_unit_labels)
        if _db.db_exists():
            self._load_material_dropdowns()

    # ── layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        root = self.root
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # ── top bar ───────────────────────────────────────────────────────────
        top = ttk.Frame(root, padding=(12, 8))
        top.grid(row=0, column=0, sticky="ew")

        tk.Label(top, text="Inverse Design", font=FONT_TITLE).grid(
            row=0, column=0, sticky="w", padx=(0, 20))

        tk.Label(top, text="Model:", font=FONT_LABEL).grid(row=0, column=1, padx=(0, 6))
        self._model_var = tk.StringVar(value="elastic")
        for i, name in enumerate(MODEL_NAMES):
            ttk.Radiobutton(top, text=name.capitalize(),
                            variable=self._model_var, value=name).grid(
                row=0, column=2 + i, padx=4)

        self._load_btn = ttk.Button(top, text="Load Model", command=self._on_load)
        self._load_btn.grid(row=0, column=4, padx=(16, 6))

        self._load_status = tk.Label(top, text="No model loaded.",
                                     font=FONT_STATUS, fg="gray")
        self._load_status.grid(row=0, column=5, padx=8, sticky="w")

        # unit system selector
        ttk.Separator(top, orient="vertical").grid(row=0, column=6,
                                                    sticky="ns", padx=12)
        tk.Label(top, text="Units:", font=FONT_LABEL).grid(row=0, column=7, padx=(0, 4))
        self._unit_sys_var = tk.StringVar(value=UM.current_system)
        self._unit_sys_cb  = ttk.Combobox(top, textvariable=self._unit_sys_var,
                                           values=UM.available_systems,
                                           state="readonly", width=36, font=FONT_SMALL)
        self._unit_sys_cb.grid(row=0, column=8, padx=4)
        self._unit_sys_cb.bind("<<ComboboxSelected>>", self._on_unit_system_change)

        # ── row 1: action buttons + material selectors + unit system ─────────
        ttk.Button(top, text="🌡  Open Thermal Inverse Solver",
                   command=self._open_thermal_inverse).grid(
            row=1, column=0, columnspan=4, sticky="w", padx=(0, 8), pady=(6, 0))

        self._load_card_btn = ttk.Button(top, text="📂  Load from Card",
                                         command=self._on_load_from_card)
        self._load_card_btn.grid(row=1, column=4, padx=(16, 0), pady=(6, 0), sticky="w")

        ttk.Separator(top, orient="vertical").grid(row=1, column=5,
                                                    sticky="ns", padx=12, pady=(6, 0))
        tk.Label(top, text="Fiber:", font=FONT_LABEL).grid(
            row=1, column=6, padx=(0, 4), pady=(6, 0))
        self._fiber_var = tk.StringVar(value="— select —")
        self._fiber_cb  = ttk.Combobox(top, textvariable=self._fiber_var,
                                        state="readonly", width=22, font=FONT_ENTRY)
        self._fiber_cb.grid(row=1, column=7, padx=4, pady=(6, 0))
        self._fiber_cb.bind("<<ComboboxSelected>>", lambda _: self._on_load_material_props(warn_no_model=False))

        tk.Label(top, text="Polymer:", font=FONT_LABEL).grid(
            row=1, column=8, padx=(8, 4), pady=(6, 0))
        self._polymer_var = tk.StringVar(value="— select —")
        self._polymer_cb  = ttk.Combobox(top, textvariable=self._polymer_var,
                                          state="readonly", width=22, font=FONT_ENTRY)
        self._polymer_cb.grid(row=1, column=9, padx=4, pady=(6, 0))
        self._polymer_cb.bind("<<ComboboxSelected>>", lambda _: self._on_load_material_props(warn_no_model=False))

        ttk.Button(top, text="Load Props",
                   command=self._on_load_material_props).grid(
            row=1, column=10, padx=(8, 4), pady=(6, 0))


        ttk.Separator(root, orient="horizontal").grid(row=1, column=0, sticky="ew")

        # ── main panels ───────────────────────────────────────────────────────
        mid = ttk.Frame(root)
        mid.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=3, minsize=700)
        mid.grid_columnconfigure(1, weight=0)
        mid.grid_columnconfigure(2, weight=2, minsize=520)

        # inputs panel
        in_outer = ttk.LabelFrame(mid, text="Input Variables  (Fixed = known value, Free = to be optimised)", padding=8)
        in_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        in_outer.grid_rowconfigure(0, weight=1)
        in_outer.grid_columnconfigure(0, weight=1)
        self._in_canvas, self._in_frame = self._scrollframe(in_outer)
        tk.Label(self._in_frame, text="Load a model to configure inputs.",
                 font=FONT_STATUS, fg="gray").grid(row=0, column=0, padx=10, pady=10)

        ttk.Separator(mid, orient="vertical").grid(row=0, column=1, sticky="ns", padx=4)

        # targets panel
        out_outer = ttk.LabelFrame(mid, text="Target Outputs  (check to include as target)", padding=8)
        out_outer.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        out_outer.grid_rowconfigure(0, weight=1)
        out_outer.grid_columnconfigure(0, weight=1)
        self._out_canvas, self._out_frame = self._scrollframe(out_outer)
        tk.Label(self._out_frame, text="Load a model to configure targets.",
                 font=FONT_STATUS, fg="gray").grid(row=0, column=0, padx=10, pady=10)

        # ── solver bar ────────────────────────────────────────────────────────
        ttk.Separator(root, orient="horizontal").grid(row=3, column=0, sticky="ew")
        sbar = ttk.Frame(root, padding=(12, 7))
        sbar.grid(row=4, column=0, sticky="ew")

        tk.Label(sbar, text="Solver:", font=FONT_BOLD).grid(row=0, column=0, padx=(0, 10))

        tk.Label(sbar, text="Method:", font=FONT_LABEL).grid(row=0, column=1, padx=(0, 4))
        self._method_var = tk.StringVar(value="lbfgsb")
        ttk.Combobox(sbar, textvariable=self._method_var,
                     values=["lbfgs", "lbfgsb", "adam",
                             "differential_evolution", "dual_annealing", "basinhopping"],
                     width=22, state="readonly").grid(row=0, column=2, padx=4)
        ttk.Button(sbar, text="ℹ", width=2,
                   command=self._show_optimizer_info).grid(row=0, column=3, padx=(2, 6))
        tk.Label(sbar, text="(global methods require bounds)",
                 font=FONT_SMALL, fg="gray").grid(row=1, column=1, columnspan=4, sticky="w", padx=(0, 4))

        tk.Label(sbar, text="Max iter:", font=FONT_LABEL).grid(row=0, column=4, padx=(14, 4))
        self._maxiter_var = tk.StringVar(value="300")
        tk.Entry(sbar, textvariable=self._maxiter_var, width=7,
                 font=FONT_ENTRY).grid(row=0, column=5, padx=4)

        tk.Label(sbar, text="Tol:", font=FONT_LABEL).grid(row=0, column=6, padx=(14, 4))
        self._tol_var = tk.StringVar(value="1e-9")
        tk.Entry(sbar, textvariable=self._tol_var, width=9,
                 font=FONT_ENTRY).grid(row=0, column=7, padx=4)

        tk.Label(sbar, text="Penalty:", font=FONT_LABEL).grid(row=0, column=8, padx=(14, 4))
        self._penalty_var = tk.StringVar(value="10000")
        tk.Entry(sbar, textvariable=self._penalty_var, width=9,
                 font=FONT_ENTRY).grid(row=0, column=9, padx=4)

        tk.Label(sbar, text="Seed:", font=FONT_LABEL).grid(row=0, column=10, padx=(14, 4))
        self._seed_var = tk.StringVar(value="42")
        tk.Entry(sbar, textvariable=self._seed_var, width=6,
                 font=FONT_ENTRY).grid(row=0, column=11, padx=4)

        self._solve_btn = ttk.Button(sbar, text="  SOLVE  ",
                                     command=self._on_solve, state="disabled")
        self._solve_btn.grid(row=0, column=12, padx=(22, 8))

        self._solve_status = tk.Label(sbar, text="", font=FONT_STATUS, fg="orange")
        self._solve_status.grid(row=0, column=13, sticky="w")

        # ε-insensitive loss options (row 1)
        self._use_eps_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sbar, text="Use ε-insensitive loss",
                        variable=self._use_eps_var).grid(
            row=1, column=5, columnspan=2, padx=(14, 4), sticky="w")
        tk.Label(sbar, text="ε scale:", font=FONT_SMALL).grid(
            row=1, column=7, padx=(10, 4), sticky="e")
        self._eps_scale_var = tk.StringVar(value="0.5")
        tk.Entry(sbar, textvariable=self._eps_scale_var, width=6,
                 font=FONT_ENTRY).grid(row=1, column=8, padx=4)

        self._export_btn = ttk.Button(sbar, text="Export Results",
                                      command=self._on_export, state="disabled")
        self._export_btn.grid(row=1, column=9, padx=(14, 4))

        self._save_card_btn = ttk.Button(sbar, text="Save to Card",
                                         command=self._on_save_to_card,
                                         state="disabled")
        self._save_card_btn.grid(row=1, column=10, padx=(4, 4))

        # ── results panel ─────────────────────────────────────────────────────
        ttk.Separator(root, orient="horizontal").grid(row=5, column=0, sticky="ew")
        res_outer = ttk.LabelFrame(root, text="Results", padding=6)
        res_outer.grid(row=6, column=0, sticky="ew", padx=8, pady=(4, 8))
        res_outer.grid_columnconfigure(0, weight=1)

        self._results_text = tk.Text(
            res_outer, height=11, font=FONT_MONO, state="disabled",
            wrap="none", bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4")
        vsb = ttk.Scrollbar(res_outer, orient="vertical",
                            command=self._results_text.yview)
        hsb = ttk.Scrollbar(res_outer, orient="horizontal",
                            command=self._results_text.xview)
        self._results_text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._results_text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

    # ── optimizer info ────────────────────────────────────────────────────────
    def _show_optimizer_info(self):
        info = (
            "Optimizer Guide\n"
            "═══════════════════════════════════════════════════\n\n"
            "★ RECOMMENDED: lbfgsb\n"
            "   L-BFGS-B — bounded quasi-Newton method.\n"
            "   Best choice for most problems: fast, gradient-\n"
            "   based, and respects explicit variable bounds.\n"
            "   Use this unless you have a specific reason not to.\n\n"
            "lbfgs\n"
            "   Unbounded L-BFGS. Similar speed to lbfgsb but\n"
            "   does not enforce bounds; bounds are handled via\n"
            "   a penalty term instead.\n\n"
            "adam\n"
            "   Stochastic gradient descent (Adam). Slower than\n"
            "   L-BFGS-B but can escape shallow local minima.\n"
            "   Useful if lbfgsb stalls at a poor solution.\n\n"
            "differential_evolution  [requires bounds]\n"
            "   Global population-based search. Thorough but\n"
            "   slow. Use when the landscape has many local\n"
            "   minima and bounds are known.\n\n"
            "dual_annealing  [requires bounds]\n"
            "   Simulated annealing variant. Global search,\n"
            "   good for highly multimodal problems.\n\n"
            "basinhopping\n"
            "   Random restarts around a local minimum.\n"
            "   Useful for escaping local optima without\n"
            "   needing explicit bounds.\n\n"
            "───────────────────────────────────────────────────\n"
            "Tip: start with lbfgsb. If the result looks\n"
            "wrong or the error is high, try differential_\n"
            "evolution (with bounds) as a global search."
        )
        messagebox.showinfo("Optimizer Information", info)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _scrollframe(self, parent) -> tuple[tk.Canvas, ttk.Frame]:
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        vsb    = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.grid(row=0, column=1, sticky="ns")
        canvas.grid(row=0, column=0, sticky="nsew")
        frame = ttk.Frame(canvas)
        win   = canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>",
                   lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))

        def _scroll(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")

        # Bind mousewheel on the canvas itself and also activate bind_all
        # whenever the pointer enters the scrollable area so child widgets
        # (labels, entries, radiobuttons) forward scroll events too.
        canvas.bind("<MouseWheel>", _scroll)

        def _bind_mousewheel(e):
            canvas.bind_all("<MouseWheel>", _scroll)

        def _unbind_mousewheel(e):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)
        frame.bind("<Enter>", _bind_mousewheel)
        frame.bind("<Leave>", _unbind_mousewheel)

        return canvas, frame

    def _write_results(self, text: str):
        self._results_text.config(state="normal")
        self._results_text.delete("1.0", tk.END)
        if text:
            self._results_text.insert(tk.END, text)
        self._results_text.config(state="disabled")

    # ── model loading ─────────────────────────────────────────────────────────
    def _on_load(self):
        self._load_btn.config(state="disabled")
        self._solve_btn.config(state="disabled")
        name = self._model_var.get()
        self._load_status.config(text=f"Loading {name}…", fg="orange")
        threading.Thread(target=self._load_worker, args=(name,), daemon=True).start()

    def _load_worker(self, name: str):
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            from core.forward import load_forward
            model = load_forward(name)
            # warm-up compile
            dummy = jnp.zeros(len(model.input_fields), dtype=jnp.float32)
            model.predict_array(dummy).block_until_ready()
            self.root.after(0, lambda: self._on_load_ok(model, name))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._on_load_err(str(exc)))

    def _on_load_ok(self, model, name: str):
        self.model = model
        self._load_status.config(
            text=f"{name.capitalize()} loaded  "
                 f"({len(model.input_fields)} inputs / {len(model.output_fields)} outputs)",
            fg="green")
        self._load_btn.config(state="normal")
        self._solve_btn.config(state="normal")
        self._rebuild_input_panel(model)
        self._rebuild_output_panel(model)

    def _on_load_err(self, msg: str):
        self._load_status.config(text="Load failed.", fg="red")
        self._load_btn.config(state="normal")
        messagebox.showerror("Load Error", msg)

    # CTE constituent input fields (thermoelastic model only shows these)
    _CTE_INPUT_FIELDS = frozenset(
        {"f_cte11", "f_cte22", "matrix_cte", "f_cte1", "f_cte2", "m_cte"})

    # ── panel builders ────────────────────────────────────────────────────────
    def _rebuild_input_panel(self, model):
        for w in self._in_frame.winfo_children():
            w.destroy()
        self._input_rows = []

        is_thermoelastic = self._model_var.get() == "thermoelastic"

        # header row
        hdr = self._in_frame
        tk.Label(hdr, text="Field", font=FONT_BOLD, width=34, anchor="e").grid(
            row=0, column=0, padx=(8, 6), pady=(2, 6))
        tk.Label(hdr, text="Mode", font=FONT_BOLD).grid(
            row=0, column=1, padx=2, pady=(2, 6))
        tk.Label(hdr, text="Value / Init Guess", font=FONT_BOLD).grid(
            row=0, column=2, padx=6, pady=(2, 6))
        tk.Label(hdr, text="Bounds  [lo – hi]  (free vars only, optional)",
                 font=FONT_BOLD).grid(row=0, column=3, sticky="w", padx=4, pady=(2, 6))

        if is_thermoelastic:
            tk.Label(hdr,
                     text="Thermoelastic inverse: only constituent CTE fields are shown.\n"
                          "Run the elastic inverse first to determine microstructure and matrix modulus.",
                     font=FONT_SMALL, fg="#c07000").grid(
                row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        row_offset = 2 if is_thermoelastic else 1
        display_fields = [
            f for f in model.input_fields
            if not is_thermoelastic or f in self._CTE_INPUT_FIELDS
        ]
        for i, field in enumerate(display_fields):
            row = InputRow(hdr, row=row_offset + i, field=field,
                           display=_label("inputs", field))
            self._input_rows.append(row)

    def _rebuild_output_panel(self, model):
        for w in self._out_frame.winfo_children():
            w.destroy()
        self._output_rows = []

        is_thermoelastic = self._model_var.get() == "thermoelastic"

        hdr = self._out_frame
        tk.Label(hdr, text="", font=FONT_BOLD).grid(row=0, column=0, padx=(8, 4))
        tk.Label(hdr, text="Output Field", font=FONT_BOLD, width=34, anchor="e").grid(
            row=0, column=1, padx=(0, 6), pady=(2, 6))
        tk.Label(hdr, text="Target Value", font=FONT_BOLD).grid(
            row=0, column=2, padx=6, pady=(2, 6))
        tk.Label(hdr, text="σ (noise, same units as target)", font=FONT_BOLD).grid(
            row=0, column=3, padx=(4, 2), pady=(2, 6), sticky="w")

        display_fields = [
            f for f in model.output_fields
            if not is_thermoelastic or f.upper().startswith("CTE")
        ]
        for i, field in enumerate(display_fields):
            row = OutputRow(hdr, row=i + 1, field=field,
                            display=_label("outputs", field))
            self._output_rows.append(row)

        n = len(model.output_fields)
        self._units_hint_lbl = tk.Label(hdr, font=FONT_SMALL, fg="gray")
        self._units_hint_lbl.grid(row=n + 1, column=0, columnspan=5,
                                   sticky="w", padx=8, pady=(8, 2))
        self._update_units_hint()
        tk.Label(hdr,
                 text=("Note: Enter the standard deviation in the \u03c3 field.  "
                       "If entering the standard deviation, remember to check the \u03b5-insensitive button."),
                 font=FONT_SMALL, fg="#c07000").grid(
            row=n + 2, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 6))

    # ── input / target collection ─────────────────────────────────────────────
    def _collect_inputs(self):
        """Returns (fixed_inputs, free_inputs, bounds_or_None, init_free_list)."""
        fixed: dict[str, float] = {}
        free:  list[str]        = []
        bounds: dict[str, tuple] = {}
        init:  list[float]      = []

        for row in self._input_rows:
            try:
                display_val = row.get_value()
            except ValueError:
                raise ValueError(f"Invalid or missing value for  '{row.display}'.")
            val = UM.from_display(row.field, display_val)
            if row.is_free():
                free.append(row.field)
                init.append(val)
                b = row.get_bounds()
                if b is not None:
                    bounds[row.field] = (
                        UM.from_display(row.field, b[0]),
                        UM.from_display(row.field, b[1]),
                    )
            else:
                fixed[row.field] = val

        # if ANY free var has bounds, fill ±_BIG for vars without explicit bounds
        if bounds and len(bounds) < len(free):
            for k in free:
                if k not in bounds:
                    bounds[k] = (-_BIG, _BIG)

        return fixed, free, bounds if bounds else None, init

    def _collect_targets(self):
        targets: dict[str, float] = {}
        sigmas:  dict[str, float] = {}
        for row in self._output_rows:
            if row.is_active():
                try:
                    disp_tgt = row.get_target()
                except ValueError:
                    raise ValueError(f"Invalid or missing target for  '{row.display}'.")
                targets[row.field] = UM.from_display(row.field, disp_tgt)
                disp_sig = row.get_sigma()
                sigmas[row.field] = (UM.from_display(row.field, disp_sig)
                                     if disp_sig != 0.0 else 0.0)
        return targets, sigmas

    # ── solve ─────────────────────────────────────────────────────────────────
    def _on_solve(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        try:
            fixed, free, bounds, init = self._collect_inputs()
            targets, sigmas = self._collect_targets()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        if not free:
            messagebox.showwarning("No free variables",
                                   "Mark at least one input as Free.")
            return
        if not targets:
            messagebox.showwarning("No targets",
                                   "Check at least one output as a target.")
            return

        try:
            solver_cfg = {
                "method":             self._method_var.get(),
                "maxiter":            int(self._maxiter_var.get()),
                "tol":                float(self._tol_var.get()),
                "constraint_penalty": float(self._penalty_var.get()),
                "seed":               int(self._seed_var.get()),
                "use_epsilon_loss":   self._use_eps_var.get(),
                "epsilon_scale":      float(self._eps_scale_var.get()),
            }
        except ValueError as e:
            messagebox.showerror("Solver config error", str(e))
            return

        tags = {}

        # ── orientation tensor PSD check ──────────────────────────────────────
        _OT_FIELDS = ("a11", "a22", "a12", "a13", "a23")
        all_inputs = dict(fixed)
        all_inputs.update({k: v for k, v in zip(free, init)})
        if all(f in all_inputs for f in _OT_FIELDS):
            a11 = all_inputs["a11"]
            a22 = all_inputs["a22"]
            a33 = 1.0 - a11 - a22
            a12 = all_inputs["a12"]
            a13 = all_inputs["a13"]
            a23 = all_inputs["a23"]
            A = np.array([
                [a11, a12, a13],
                [a12, a22, a23],
                [a13, a23, a33],
            ])
            eigvals = np.linalg.eigvalsh(A)
            if np.any(eigvals < -1e-8):
                messagebox.showerror(
                    "Invalid Orientation Tensor",
                    "The orientation tensor (from fixed values and initial guesses) "
                    "is not positive semi-definite.\n\n"
                    f"Diagonal terms: A11={a11}, A22={a22}, A33={a33:.6g}\n"
                    f"Computed eigenvalues: {eigvals[0]:.6g}, {eigvals[1]:.6g}, {eigvals[2]:.6g}\n\n"
                    "Please enter a valid positive semi-definite orientation tensor.",
                )
                return

        self._solve_btn.config(state="disabled")
        self._solve_status.config(text="Solving…", fg="orange")
        self._write_results("")

        threading.Thread(
            target=self._solve_worker,
            args=(fixed, free, bounds, init, targets, sigmas, solver_cfg, tags),
            daemon=True,
        ).start()

    def _solve_worker(self, fixed_inputs, free_inputs, bounds, init_free,
                      target_outputs, sigmas, solver_cfg, tags):
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp

            from core.inverse import (
                InverseProblem,
                _solve,
                _assemble_x,
                make_orientation_sum_constraint,
            )

            model = self.model

            constraints = []
            c = make_orientation_sum_constraint(free_inputs)
            if c is not None:
                constraints.append(c)

            prob = InverseProblem(
                fixed_inputs   = fixed_inputs,
                free_inputs    = free_inputs,
                target_outputs = target_outputs,
                constraints    = tuple(constraints),
            )

            method        = solver_cfg["method"]
            penalty       = float(solver_cfg["constraint_penalty"])
            use_eps_loss  = bool(solver_cfg.get("use_epsilon_loss", False))
            epsilon_scale = float(solver_cfg.get("epsilon_scale", 1.0))
            init64        = jnp.array(init_free, jnp.float64)

            sigmas_list = [
                float(sigmas.get(k, 0.0)) * epsilon_scale
                for k in target_outputs.keys()
            ]

            free_vec, final_err = _solve(
                model.predict_array, prob,
                model.in_idx, model.out_idx, len(model.input_fields),
                model.output_std,
                init64, bounds, method, penalty,
                sigmas_list  = sigmas_list,
                use_eps_loss = use_eps_loss,
                maxiter = int(solver_cfg["maxiter"]),
                tol     = float(solver_cfg["tol"]),
                seed    = int(solver_cfg.get("seed", 42)),
            )

            x_star = _assemble_x(free_vec, prob, model.in_idx, len(model.input_fields))
            y_star = model.predict_array(x_star)
            y_np   = np.asarray(y_star)

            opt_free = {k: float(v) for k, v in zip(free_inputs, free_vec)}

            result = {
                "timestamp":              datetime.datetime.now().isoformat(timespec="seconds"),
                "tags":                   tags,
                "model":                  self._model_var.get(),
                "free_variables":         opt_free,
                "fixed_inputs":           fixed_inputs,
                "bounds":                 {k: list(v) for k, v in (bounds or {}).items()},
                "predicted_outputs":      {k: float(y_np[model.out_idx[k]])
                                           for k in model.output_fields},
                "target_outputs":         target_outputs,
                "sigmas":                 sigmas,
                "final_optimiser_error":  final_err,
                "solver":                 solver_cfg,
            }

            self.root.after(
                0,
                lambda r=result, o=opt_free, t=target_outputs, y=y_np:
                    self._on_solve_ok(r, o, t, y),
            )

        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.root.after(0, lambda m=str(exc), t=tb: self._on_solve_err(m, t))

    def _on_solve_ok(self, result: dict, opt_free: dict,
                     targets: dict, y_np: np.ndarray):
        model = self.model

        # ── format result text ────────────────────────────────────────────────
        W = 74
        lines = [
            "=" * W,
            "  INVERSE DESIGN RESULT",
        ]
        if result["tags"]:
            lines.append(f"  Tags : {result['tags']}")
        lines += [
            f"  Model: {result['model']}     Timestamp: {result['timestamp']}",
            "=" * W,
            "",
            f"Optimised free variables  ({UM.current_system}):",
        ]
        for k, v in opt_free.items():
            disp_v = UM.to_display(k, v)
            unit   = UM.unit_label(k)
            suffix = f"  {unit}" if unit else ""
            lines.append(f"  {k:<30s}: {disp_v:.8g}{suffix}")
        if "a11" in opt_free and "a22" in opt_free:
            s = opt_free["a11"] + opt_free["a22"]
            lines.append(f"  {'a11 + a22':<30s}: {s:.8g}   [constraint: <= 1.0]")

        lines += [
            "",
            f"Final optimiser error : {result['final_optimiser_error']:.4e}",
            "",
            "Predicted vs Target  (display units):",
            f"  {'Field':<12}  {'Predicted':>16}  {'Target':>16}  {'%Err':>9}  Unit",
            "  " + "-" * 62,
        ]
        for k, tgt in targets.items():
            pred   = float(y_np[model.out_idx[k]])
            unit   = UM.unit_label(k)
            p_disp = UM.to_display(k, pred)
            t_disp = UM.to_display(k, tgt)
            pct    = 100.0 * abs(pred - tgt) / tgt if tgt != 0.0 else float("nan")
            lines.append(
                f"  {k:<12}  {p_disp:>16.6g}  {t_disp:>16.6g}"
                f"  {pct:>8.3f}%  {unit}"
            )

        lines += ["", f"All predicted outputs  ({UM.current_system}):"]
        for k in model.output_fields:
            pred   = float(y_np[model.out_idx[k]])
            unit   = UM.unit_label(k)
            p_disp = UM.to_display(k, pred)
            mark   = "  <-- target" if k in targets else ""
            lines.append(f"  {k:<12}  {p_disp:>16.6g}  {unit}{mark}")

        lines.append("")
        self._write_results("\n".join(lines))

        self._last_result = result
        self._export_btn.config(state="normal")
        self._save_card_btn.config(state="normal")
        self._solve_status.config(text="Done.", fg="green")
        self._solve_btn.config(state="normal")

    def _on_solve_err(self, msg: str, tb: str):
        self._solve_status.config(text="Failed.", fg="red")
        self._solve_btn.config(state="normal")
        self._write_results(f"ERROR:\n{msg}\n\nTraceback:\n{tb}")
        messagebox.showerror("Solve Error", msg)

    # ── material card save / load ──────────────────────────────────────────────

    def _on_save_to_card(self):
        if self._last_result is None:
            messagebox.showwarning("No results", "Run the solver first.")
            return
        if not _db.db_exists():
            messagebox.showwarning("No database",
                                   "Run  python init_db.py  first.")
            return
        from gui_card_dialogs import SaveToCardDialog
        SaveToCardDialog(self.root, self._last_result,
                         fiber_id=self._card_fiber_id,
                         polymer_id=self._card_polymer_id)

    def _on_load_from_card(self):
        if not _db.db_exists():
            messagebox.showwarning("No database",
                                   "Run  python init_db.py  first.")
            return
        from gui_card_dialogs import LoadFromCardDialog
        dlg = LoadFromCardDialog(self.root)
        if not dlg.loaded:
            return
        for row in self._input_rows:
            if row.field in dlg.loaded:
                row.set_value(UM.to_display(row.field, dlg.loaded[row.field]))
        # remember which fiber/polymer this card belongs to
        if dlg.loaded_card is not None:
            self._card_fiber_id   = dlg.loaded_card.get("fiber_id")
            self._card_polymer_id = dlg.loaded_card.get("polymer_id")
            # sync fiber/polymer dropdowns
            fid = self._card_fiber_id
            pid = self._card_polymer_id
            if fid is not None:
                id_to_name = {v: k for k, v in self._fiber_map.items()}
                name = id_to_name.get(fid)
                if name:
                    self._fiber_var.set(name)
            if pid is not None:
                id_to_name = {v: k for k, v in self._polymer_map.items()}
                name = id_to_name.get(pid)
                if name:
                    self._polymer_var.set(name)

    def _load_material_dropdowns(self):
        try:
            fibers   = _db.get_all_fibers()
            polymers = _db.get_all_polymers()
        except Exception:
            return
        self._fiber_map   = {f["name"]: f["id"] for f in fibers}
        self._polymer_map = {p["name"]: p["id"] for p in polymers}
        self._fiber_cb["values"]   = list(self._fiber_map)
        self._polymer_cb["values"] = list(self._polymer_map)

    def _on_load_material_props(self, *, warn_no_model: bool = True):
        if not _db.db_exists():
            messagebox.showwarning("No database", "Run  python init_db.py  first.")
            return
        if not self._fiber_map and not self._polymer_map:
            self._load_material_dropdowns()

        fid = self._fiber_map.get(self._fiber_var.get())
        pid = self._polymer_map.get(self._polymer_var.get())

        if fid is None and pid is None:
            messagebox.showwarning("No material selected",
                                   "Select a fiber and/or polymer first.")
            return
        if not self._input_rows:
            if warn_no_model:
                messagebox.showwarning("No model loaded",
                                       "Load a model first so input fields are available.")
            return

        filled = 0
        if fid is not None:
            try:
                props = _db.fiber_model_inputs(fid)
                for row in self._input_rows:
                    if row.field in props:
                        row.set_value(UM.to_display(row.field, props[row.field]))
                        filled += 1
            except Exception as exc:
                messagebox.showerror("Error", f"Could not load fiber properties:\n{exc}")
                return

        if pid is not None:
            try:
                props = _db.polymer_model_inputs(pid)
                for row in self._input_rows:
                    if row.field in props:
                        row.set_value(UM.to_display(row.field, props[row.field]))
                        filled += 1
            except Exception as exc:
                messagebox.showerror("Error", f"Could not load polymer properties:\n{exc}")
                return

        self._card_fiber_id   = fid
        self._card_polymer_id = pid

    # ── unit system ───────────────────────────────────────────────────────────

    def _on_unit_system_change(self, _event=None):
        new_sys = self._unit_sys_var.get()
        old_sys = UM.current_system
        if new_sys == old_sys:
            return
        # Convert existing values and bounds in input rows
        for row in self._input_rows:
            try:
                old_val = row.get_value()
            except ValueError:
                continue
            row.set_value(UM.convert_between(row.field, old_val, old_sys, new_sys))
            row.convert_bounds(old_sys, new_sys)
        # Convert active target / sigma values in output rows
        for row in self._output_rows:
            if row.is_active():
                try:
                    old_tgt = row.get_target()
                    row.set_target(UM.convert_between(row.field, old_tgt, old_sys, new_sys))
                except ValueError:
                    pass
                old_sig = row.get_sigma()
                if old_sig != 0.0:
                    row.set_sigma(UM.convert_between(row.field, old_sig, old_sys, new_sys))
        UM.set_system(new_sys)   # fires UM callbacks → _refresh_unit_labels

    def _refresh_unit_labels(self):
        """Called by UM when any window changes the unit system."""
        if hasattr(self, "_unit_sys_var"):
            self._unit_sys_var.set(UM.current_system)
        for row in self._input_rows:
            row.update_unit_label()
        for row in self._output_rows:
            row.update_unit_label()
        self._update_units_hint()

    def _update_units_hint(self):
        if not hasattr(self, "_units_hint_lbl"):
            return
        mod_u = UM.unit_label("E1") or "MPa"
        cte_u = UM.unit_label("CTE11") or "1/K"
        self._units_hint_lbl.config(
            text=f"Unit system: {UM.current_system}  ·  "
                 f"E/G in {mod_u}  ·  CTE in {cte_u}  ·  \u03bd dimensionless")

    def _open_thermal_inverse(self):
        from gui_thermal_inverse import ThermalInverseWindow
        win = ThermalInverseWindow(parent=self.root)
        win._win.lift()
        win._win.focus_force()
        win._win.attributes("-topmost", True)
        win._win.after(1500, lambda: win._win.attributes("-topmost", False))

    def _on_export(self):
        if self._last_result is None:
            messagebox.showwarning("No results", "Run the solver first.")
            return

        model_name   = self._last_result.get("model", "unknown")
        ts           = self._last_result.get("timestamp", "")[:10]
        default_name = f"inverse_{model_name}_{ts}.json"
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=default_name,
            title="Export Inverse Result",
        )
        if not path:
            return

        try:
            with open(path, "w") as f:
                json.dump(self._last_result, f, indent=2)
            self._solve_status.config(
                text=f"Saved → {fname}", fg="green")
        except Exception as exc:
            messagebox.showwarning("Save warning", f"Could not save result JSON:\n{exc}")


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    import subprocess
    import sys

    root = tk.Tk()
    InverseGUI(root)

    def _bring_to_front():
        root.lift()
        root.focus_force()
        root.attributes("-topmost", True)
        root.after(1500, lambda: root.attributes("-topmost", False))

    # On macOS, use AppleScript to activate the app so it surfaces even when
    # the terminal is in fullscreen / another Mission Control space.
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["osascript", "-e",
                 f'tell application "System Events" to set frontmost of '
                 f'every process whose unix id is {os.getpid()} to true'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    root.after(100, _bring_to_front)
    root.mainloop()


if __name__ == "__main__":
    main()
