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
import sys
import threading
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_EL_DIR = os.path.normpath(os.path.join(_HERE, "..", "EL_surrogate"))
if _EL_DIR not in sys.path:
    sys.path.insert(0, _EL_DIR)

os.environ.setdefault("JAX_ENABLE_X64",    "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

# ── field label mapping ────────────────────────────────────────────────────────
_LABELS_FILE = os.path.join(_HERE, "field_labels.json")

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


# ── output display scales and units ───────────────────────────────────────────
OUTPUT_SCALES: dict[str, float] = {
    "E1":  1e-3,  "E2":  1e-3,  "E3":  1e-3,
    "G12": 1e-3,  "G13": 1e-3,  "G23": 1e-3,
    "nu12": 1.0,  "nu13": 1.0,  "nu23": 1.0,
    "CTE11": 1e6, "CTE22": 1e6, "CTE33": 1e6,
    "CTE12": 1e6, "CTE13": 1e6, "CTE23": 1e6,
}
OUTPUT_UNITS: dict[str, str] = {
    "E1": "GPa",  "E2": "GPa",  "E3": "GPa",
    "G12": "GPa", "G13": "GPa", "G23": "GPa",
    "nu12": "",   "nu13": "",   "nu23": "",
    "CTE11": "µ/K", "CTE22": "µ/K", "CTE33": "µ/K",
    "CTE12": "µ/K", "CTE13": "µ/K", "CTE23": "µ/K",
}
# Units used for target/sigma *entry* (raw model units, no scaling)
ENTRY_UNITS: dict[str, str] = {
    "E1": "MPa",  "E2": "MPa",  "E3": "MPa",
    "G12": "MPa", "G13": "MPa", "G23": "MPa",
    "nu12": "",   "nu13": "",   "nu23": "",
    "CTE11": "1/K", "CTE22": "1/K", "CTE33": "1/K",
    "CTE12": "1/K", "CTE13": "1/K", "CTE23": "1/K",
}

FONT_TITLE  = ("Helvetica", 17, "bold")
FONT_LABEL  = ("Helvetica", 14)
FONT_BOLD   = ("Helvetica", 14, "bold")
FONT_SMALL  = ("Helvetica", 11)
FONT_STATUS = ("Helvetica", 13, "italic")
FONT_ENTRY  = ("Helvetica", 13)
FONT_MONO   = ("Courier",   12)

MODEL_NAMES = ["elastic", "thermoelastic"]
_BIG        = 1e12   # stand-in for ±∞ when only some free vars have explicit bounds


# ── tag parsing ────────────────────────────────────────────────────────────────
def parse_tags(tag_str: str) -> dict:
    """Parse "key:value key2:value2" into a dict.  "machine:CAMRI" -> {"machine":"CAMRI"}"""
    result = {}
    for part in tag_str.strip().split():
        if ":" in part:
            k, _, v = part.partition(":")
            result[k.strip()] = v.strip()
        elif part:
            result[part] = ""
    return result


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

        # value entry (fixed value or initial guess for free)
        self._val_entry = tk.Entry(parent, width=13, font=FONT_ENTRY)
        self._val_entry.grid(row=row, column=2, padx=6, pady=2)

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
        self._val_entry.insert(0, str(v))


# ── per-output row ─────────────────────────────────────────────────────────────
class OutputRow:
    """One row in the Targets panel (one model output field)."""

    def __init__(self, parent: tk.Widget, row: int, field: str, display: str, unit: str = ""):
        self.field   = field
        self.display = display
        self._active = tk.BooleanVar(value=False)

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
        if unit:
            tk.Label(sigma_frame, text=unit, font=FONT_SMALL, fg="gray").pack(side="left", padx=(3, 0))

        unit_col_text = f"({unit})" if unit else "(dimensionless)"
        tk.Label(parent, text=unit_col_text, font=FONT_SMALL,
                 fg="gray").grid(row=row, column=4, sticky="w", padx=2)

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


# ── main GUI ───────────────────────────────────────────────────────────────────
class InverseGUI:

    def __init__(self, root: tk.Tk):
        self.root         = root
        self.model        = None
        self._input_rows:  list[InputRow]  = []
        self._output_rows: list[OutputRow] = []
        self._last_result: Optional[dict]  = None

        root.title("Composite Surrogate – Inverse Design")
        root.geometry("1700x1020")
        root.minsize(1200, 700)
        self._build_layout()

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

        # tag field
        ttk.Separator(top, orient="vertical").grid(row=0, column=6,
                                                    sticky="ns", padx=12)
        tk.Label(top, text="Tag:", font=FONT_LABEL).grid(row=0, column=7, padx=(0, 4))
        self._tag_var = tk.StringVar()
        tk.Entry(top, textvariable=self._tag_var, width=24,
                 font=FONT_ENTRY).grid(row=0, column=8, padx=4)
        tk.Label(top, text='e.g.  machine:CAMRI  batch:A',
                 font=FONT_SMALL, fg="gray").grid(row=0, column=9, padx=4)

        # save directory
        ttk.Separator(top, orient="vertical").grid(row=0, column=10,
                                                    sticky="ns", padx=12)
        tk.Label(top, text="Save dir:", font=FONT_LABEL).grid(row=0, column=11, padx=(0, 4))
        self._savedir_var = tk.StringVar(value=_HERE)
        tk.Entry(top, textvariable=self._savedir_var, width=30,
                 font=FONT_ENTRY).grid(row=0, column=12, padx=4)
        ttk.Button(top, text="Browse…",
                   command=self._browse_savedir).grid(row=0, column=13, padx=4)

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
        self._method_var = tk.StringVar(value="lbfgs")
        ttk.Combobox(sbar, textvariable=self._method_var,
                     values=["lbfgs", "lbfgsb", "adam",
                             "differential_evolution", "dual_annealing", "basinhopping"],
                     width=22, state="readonly").grid(row=0, column=2, padx=4)
        tk.Label(sbar, text="(global methods require bounds)",
                 font=FONT_SMALL, fg="gray").grid(row=1, column=1, columnspan=3, sticky="w", padx=(0, 4))

        tk.Label(sbar, text="Max iter:", font=FONT_LABEL).grid(row=0, column=3, padx=(14, 4))
        self._maxiter_var = tk.StringVar(value="300")
        tk.Entry(sbar, textvariable=self._maxiter_var, width=7,
                 font=FONT_ENTRY).grid(row=0, column=4, padx=4)

        tk.Label(sbar, text="Tol:", font=FONT_LABEL).grid(row=0, column=5, padx=(14, 4))
        self._tol_var = tk.StringVar(value="1e-9")
        tk.Entry(sbar, textvariable=self._tol_var, width=9,
                 font=FONT_ENTRY).grid(row=0, column=6, padx=4)

        tk.Label(sbar, text="Penalty:", font=FONT_LABEL).grid(row=0, column=7, padx=(14, 4))
        self._penalty_var = tk.StringVar(value="10000")
        tk.Entry(sbar, textvariable=self._penalty_var, width=9,
                 font=FONT_ENTRY).grid(row=0, column=8, padx=4)

        tk.Label(sbar, text="Seed:", font=FONT_LABEL).grid(row=0, column=9, padx=(14, 4))
        self._seed_var = tk.StringVar(value="42")
        tk.Entry(sbar, textvariable=self._seed_var, width=6,
                 font=FONT_ENTRY).grid(row=0, column=10, padx=4)

        self._solve_btn = ttk.Button(sbar, text="  SOLVE  ",
                                     command=self._on_solve, state="disabled")
        self._solve_btn.grid(row=0, column=11, padx=(22, 8))
        self._export_btn = ttk.Button(sbar, text="Export Results",
                                      command=self._on_export, state="disabled")
        self._export_btn.grid(row=0, column=12, padx=(4, 8))
        self._solve_status = tk.Label(sbar, text="", font=FONT_STATUS, fg="orange")
        self._solve_status.grid(row=0, column=13, sticky="w")

        # ε-insensitive loss options (row 1)
        self._use_eps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sbar, text="Use ε-insensitive loss",
                        variable=self._use_eps_var).grid(
            row=1, column=4, columnspan=2, padx=(14, 4), sticky="w")
        tk.Label(sbar, text="ε scale:", font=FONT_SMALL).grid(
            row=1, column=6, padx=(10, 4), sticky="e")
        self._eps_scale_var = tk.StringVar(value="1.0")
        tk.Entry(sbar, textvariable=self._eps_scale_var, width=6,
                 font=FONT_ENTRY).grid(row=1, column=7, padx=4)

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
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        return canvas, frame

    def _browse_savedir(self):
        d = filedialog.askdirectory(initialdir=self._savedir_var.get())
        if d:
            self._savedir_var.set(d)

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
            from forward import load_forward
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

    # ── panel builders ────────────────────────────────────────────────────────
    def _rebuild_input_panel(self, model):
        for w in self._in_frame.winfo_children():
            w.destroy()
        self._input_rows = []

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

        for i, field in enumerate(model.input_fields):
            row = InputRow(hdr, row=i + 1, field=field,
                           display=_label("inputs", field))
            self._input_rows.append(row)

    def _rebuild_output_panel(self, model):
        for w in self._out_frame.winfo_children():
            w.destroy()
        self._output_rows = []

        hdr = self._out_frame
        tk.Label(hdr, text="", font=FONT_BOLD).grid(row=0, column=0, padx=(8, 4))
        tk.Label(hdr, text="Output Field", font=FONT_BOLD, width=34, anchor="e").grid(
            row=0, column=1, padx=(0, 6), pady=(2, 6))
        tk.Label(hdr, text="Target Value", font=FONT_BOLD).grid(
            row=0, column=2, padx=6, pady=(2, 6))
        tk.Label(hdr, text="σ (noise, same units as target)", font=FONT_BOLD).grid(
            row=0, column=3, padx=(4, 2), pady=(2, 6), sticky="w")

        for i, field in enumerate(model.output_fields):
            row = OutputRow(hdr, row=i + 1, field=field,
                            display=_label("outputs", field),
                            unit=ENTRY_UNITS.get(field, ""))
            self._output_rows.append(row)

        n = len(model.output_fields)
        tk.Label(hdr,
                 text="Units for target entry: E/G in MPa · nu dimensionless · CTE in 1/K",
                 font=FONT_SMALL, fg="gray").grid(
            row=n + 1, column=0, columnspan=5, sticky="w", padx=8, pady=(8, 2))

    # ── input / target collection ─────────────────────────────────────────────
    def _collect_inputs(self):
        """Returns (fixed_inputs, free_inputs, bounds_or_None, init_free_list)."""
        fixed: dict[str, float] = {}
        free:  list[str]        = []
        bounds: dict[str, tuple] = {}
        init:  list[float]      = []

        for row in self._input_rows:
            try:
                val = row.get_value()
            except ValueError:
                raise ValueError(f"Invalid or missing value for  '{row.display}'.")
            if row.is_free():
                free.append(row.field)
                init.append(val)
                b = row.get_bounds()
                if b is not None:
                    bounds[row.field] = b
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
                    targets[row.field] = row.get_target()
                except ValueError:
                    raise ValueError(f"Invalid or missing target for  '{row.display}'.")
                sigmas[row.field] = row.get_sigma()
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

        tag_str = self._tag_var.get().strip()
        tags    = parse_tags(tag_str) if tag_str else {}

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

            from inverse import (
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
            "Optimised free variables:",
        ]
        for k, v in opt_free.items():
            lines.append(f"  {k:<30s}: {v:.8g}")
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
            pred  = float(y_np[model.out_idx[k]])
            scale = OUTPUT_SCALES.get(k, 1.0)
            unit  = OUTPUT_UNITS.get(k, "")
            pct   = 100.0 * abs(pred - tgt) / tgt if tgt != 0.0 else float("nan")
            lines.append(
                f"  {k:<12}  {pred * scale:>16.6g}  {tgt * scale:>16.6g}"
                f"  {pct:>8.3f}%  {unit}"
            )

        lines += ["", "All predicted outputs  (display units):"]
        for k in model.output_fields:
            pred  = float(y_np[model.out_idx[k]])
            scale = OUTPUT_SCALES.get(k, 1.0)
            unit  = OUTPUT_UNITS.get(k, "")
            mark  = "  <-- target" if k in targets else ""
            lines.append(f"  {k:<12}  {pred * scale:>16.6g}  {unit}{mark}")

        lines.append("")
        self._write_results("\n".join(lines))

        self._last_result = result
        self._export_btn.config(state="normal")
        self._solve_status.config(text="Done.", fg="green")
        self._solve_btn.config(state="normal")

    def _on_solve_err(self, msg: str, tb: str):
        self._solve_status.config(text="Failed.", fg="red")
        self._solve_btn.config(state="normal")
        self._write_results(f"ERROR:\n{msg}\n\nTraceback:\n{tb}")
        messagebox.showerror("Solve Error", msg)

    def _on_export(self):
        if self._last_result is None:
            messagebox.showwarning("No results", "Run the solver first.")
            return

        tags = self._last_result.get("tags", {})
        name_tag = tags.get("name", "").strip()
        if not name_tag:
            messagebox.showerror("Export error", "name tag missing")
            return

        savedir = self._savedir_var.get().strip()
        if not savedir:
            messagebox.showwarning("No save directory", "Set a save directory first.")
            return

        try:
            os.makedirs(savedir, exist_ok=True)
        except Exception as exc:
            messagebox.showwarning("Save warning", f"Could not create directory:\n{exc}")
            return

        model_name = self._last_result.get("model", "unknown")
        fname = f"inverse_{model_name}_{name_tag}.json"
        path  = os.path.join(savedir, fname)

        if os.path.exists(path):
            overwrite = messagebox.askyesno(
                "File exists",
                f"A result with name tag '{name_tag}' already exists:\n{fname}\n\nOverwrite?",
            )
            if not overwrite:
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
    try:
        tmp = tk.Tk()
        tmp.destroy()
    except tk.TclError:
        pass
    root = tk.Tk()
    InverseGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
