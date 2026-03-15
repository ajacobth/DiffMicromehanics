"""gui.py – unified surrogate GUI for forward evaluation.

Supports both the elastic and thermoelastic surrogates. Select the model,
click Load, fill in the inputs, then click Predict.

Usage
-----
    python gui.py

Requires that model artifacts have been exported into models/{elastic,thermoelastic}/.
Run the export scripts in the training folders first if those folders are empty.

Field labels
------------
Edit field_labels.json to customise how input/output names appear in the GUI.
Any field not listed in that file falls back to its raw internal name.
"""
from __future__ import annotations

import json
import os
import threading

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

# ── field label mapping ───────────────────────────────────────────────────────
_LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "field_labels.json")

def _load_field_labels() -> dict:
    """Load field_labels.json if present; silently return empty dict on failure."""
    try:
        with open(_LABELS_FILE) as f:
            data = json.load(f)
        # strip comment key
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

_FIELD_LABELS = _load_field_labels()

def _label(section: str, key: str) -> str:
    """Return the display name for a field, falling back to the raw key."""
    return _FIELD_LABELS.get(section, {}).get(key, key)


# ── output scale factors (raw model outputs are MPa / 1/K) ───────────────────
OUTPUT_SCALES: dict[str, float] = {
    "E1":  1e-3, "E2":  1e-3, "E3":  1e-3,
    "G12": 1e-3, "G13": 1e-3, "G23": 1e-3,
    "nu12": 1.0, "nu13": 1.0, "nu23": 1.0,
    "CTE11": 1e6, "CTE22": 1e6, "CTE33": 1e6,
    "CTE12": 1e6, "CTE13": 1e6, "CTE23": 1e6,
}

FONT_TITLE  = ("Helvetica", 17, "bold")
FONT_LABEL  = ("Helvetica", 15)
FONT_BOLD   = ("Helvetica", 15, "bold")
FONT_STATUS = ("Helvetica", 13, "italic")
FONT_ENTRY  = ("Helvetica", 15)

MODEL_NAMES = ["elastic", "thermoelastic"]


# ── main GUI class ────────────────────────────────────────────────────────────
class SurrogateGUI:
    """Unified forward-evaluation GUI for elastic and thermoelastic surrogates."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Composite Micromechanics Surrogate")
        root.geometry("1500x960")
        root.minsize(1100, 600)

        self.model:         object | None = None
        self.input_entries: dict[str, tk.Entry] = {}
        self.output_labels: dict[str, tk.Label] = {}

        self._build_layout()

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_layout(self):
        root = self.root
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # ── top bar ──────────────────────────────────────────────────────────
        top = ttk.Frame(root, padding=(12, 10))
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(5, weight=1)

        tk.Label(top, text="Composite Surrogate Predictor",
                 font=FONT_TITLE).grid(row=0, column=0, sticky="w", padx=(0, 24))

        tk.Label(top, text="Model:", font=FONT_LABEL).grid(row=0, column=1, padx=(0, 6))

        self._model_var = tk.StringVar(value="elastic")
        for i, name in enumerate(MODEL_NAMES):
            ttk.Radiobutton(top, text=name.capitalize(), variable=self._model_var,
                            value=name).grid(row=0, column=2 + i, padx=6)

        self._load_btn = ttk.Button(top, text="Load Model", command=self._on_load)
        self._load_btn.grid(row=0, column=4, padx=(18, 6), sticky="w")

        self._load_status = tk.Label(top, text="No model loaded.",
                                     font=FONT_STATUS, fg="gray")
        self._load_status.grid(row=0, column=5, padx=10, sticky="w")

        ttk.Button(top, text="Identifiability Check",
                   command=self._open_identifiability).grid(
            row=0, column=6, padx=(12, 4), sticky="e"
        )

        ttk.Separator(root, orient="horizontal").grid(row=0, column=0, sticky="ew")

        # ── main area: inputs (left) + outputs (right) ───────────────────────
        mid = ttk.Frame(root)
        mid.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        mid.grid_rowconfigure(0, weight=1)
        mid.grid_columnconfigure(0, weight=1, minsize=520)
        mid.grid_columnconfigure(1, weight=0)
        mid.grid_columnconfigure(2, weight=1, minsize=520)

        # ── inputs panel ─────────────────────────────────────────────────────
        in_outer = ttk.LabelFrame(mid, text="Inputs", padding=12)
        in_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        in_outer.grid_rowconfigure(0, weight=1)
        in_outer.grid_columnconfigure(0, weight=1)

        # scrollable canvas so panel works even on small screens
        self._in_canvas = tk.Canvas(in_outer, borderwidth=0, highlightthickness=0)
        in_vsb = ttk.Scrollbar(in_outer, orient="vertical",
                                command=self._in_canvas.yview)
        self._in_canvas.configure(yscrollcommand=in_vsb.set)
        in_vsb.grid(row=0, column=1, sticky="ns")
        self._in_canvas.grid(row=0, column=0, sticky="nsew")

        self._in_frame = ttk.Frame(self._in_canvas)
        self._in_canvas_win = self._in_canvas.create_window(
            (0, 0), window=self._in_frame, anchor="nw")
        self._in_frame.bind("<Configure>", self._on_in_frame_configure)
        self._in_canvas.bind("<Configure>", self._on_in_canvas_configure)
        self._in_canvas.bind("<MouseWheel>",
            lambda e: self._in_canvas.yview_scroll(int(-e.delta / 120), "units"))

        # predict button row below the canvas
        ctrl = ttk.Frame(in_outer)
        ctrl.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self._predict_btn = ttk.Button(ctrl, text="Predict",
                                       command=self._predict_start, state="disabled")
        self._predict_btn.pack(side="left", padx=4)
        self._pred_status = tk.Label(ctrl, text="", font=FONT_STATUS, fg="orange")
        self._pred_status.pack(side="left", padx=10)

        # separator
        ttk.Separator(mid, orient="vertical").grid(row=0, column=1,
                                                    sticky="ns", padx=4)

        # ── outputs panel ────────────────────────────────────────────────────
        out_outer = ttk.LabelFrame(mid, text="Outputs", padding=12)
        out_outer.grid(row=0, column=2, sticky="nsew", padx=(4, 0))
        out_outer.grid_rowconfigure(0, weight=1)
        out_outer.grid_columnconfigure(0, weight=1)

        self._out_canvas = tk.Canvas(out_outer, borderwidth=0, highlightthickness=0)
        out_vsb = ttk.Scrollbar(out_outer, orient="vertical",
                                 command=self._out_canvas.yview)
        self._out_canvas.configure(yscrollcommand=out_vsb.set)
        out_vsb.grid(row=0, column=1, sticky="ns")
        self._out_canvas.grid(row=0, column=0, sticky="nsew")

        self._out_frame = ttk.Frame(self._out_canvas)
        self._out_canvas_win = self._out_canvas.create_window(
            (0, 0), window=self._out_frame, anchor="nw")
        self._out_frame.bind("<Configure>", self._on_out_frame_configure)
        self._out_canvas.bind("<Configure>", self._on_out_canvas_configure)
        self._out_canvas.bind("<MouseWheel>",
            lambda e: self._out_canvas.yview_scroll(int(-e.delta / 120), "units"))

        # placeholder text
        self._in_placeholder = tk.Label(self._in_frame,
                                        text="Select a model and click Load.",
                                        font=FONT_STATUS, fg="gray")
        self._in_placeholder.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self._out_placeholder = tk.Label(self._out_frame,
                                         text="Predictions will appear here.",
                                         font=FONT_STATUS, fg="gray")
        self._out_placeholder.grid(row=0, column=0, padx=10, pady=10, sticky="w")

    # ── canvas resize helpers ─────────────────────────────────────────────────
    def _on_in_frame_configure(self, _event=None):
        self._in_canvas.configure(scrollregion=self._in_canvas.bbox("all"))

    def _on_in_canvas_configure(self, event):
        self._in_canvas.itemconfig(self._in_canvas_win, width=event.width)

    def _on_out_frame_configure(self, _event=None):
        self._out_canvas.configure(scrollregion=self._out_canvas.bbox("all"))

    def _on_out_canvas_configure(self, event):
        self._out_canvas.itemconfig(self._out_canvas_win, width=event.width)

    # ── model loading ─────────────────────────────────────────────────────────
    def _on_load(self):
        self._load_btn.config(state="disabled")
        self._predict_btn.config(state="disabled")
        model_name = self._model_var.get()
        self._load_status.config(text=f"Loading {model_name}…", fg="orange")
        threading.Thread(target=self._load_worker,
                         args=(model_name,), daemon=True).start()

    def _load_worker(self, model_name: str):
        try:
            import jax.numpy as jnp
            from forward import load_forward
            model = load_forward(model_name)
            dummy = jnp.zeros(len(model.input_fields), dtype=jnp.float32)
            model.predict_array(dummy).block_until_ready()
            self.root.after(0, lambda: self._on_load_success(model, model_name))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._on_load_error(str(exc)))

    def _on_load_success(self, model, model_name: str):
        self.model = model
        self._load_status.config(
            text=f"{model_name.capitalize()} loaded  "
                 f"({len(model.input_fields)} in / {len(model.output_fields)} out)",
            fg="green",
        )
        self._load_btn.config(state="normal")
        self._predict_btn.config(state="normal")
        self._rebuild_input_panel(model)
        self._rebuild_output_panel(model)

    def _on_load_error(self, msg: str):
        self._load_status.config(text="Load failed – see error.", fg="red")
        self._load_btn.config(state="normal")
        messagebox.showerror("Load Error", msg)

    # ── dynamic panel builders ────────────────────────────────────────────────
    def _rebuild_input_panel(self, model):
        for w in self._in_frame.winfo_children():
            w.destroy()
        self.input_entries = {}
        self._in_frame.grid_columnconfigure(1, weight=1)

        for i, name in enumerate(model.input_fields):
            display = _label("inputs", name)
            tk.Label(self._in_frame, text=f"{display}:", font=FONT_LABEL,
                     anchor="e").grid(row=i, column=0, sticky="e",
                                      padx=(8, 10), pady=5)
            ent = tk.Entry(self._in_frame, width=20, font=FONT_ENTRY)
            ent.grid(row=i, column=1, sticky="ew", padx=(0, 12), pady=5)
            ent.bind("<Return>", lambda _e: self._predict_start())
            self.input_entries[name] = ent

    def _rebuild_output_panel(self, model):
        for w in self._out_frame.winfo_children():
            w.destroy()
        self.output_labels = {}
        self._out_frame.grid_columnconfigure(1, weight=1)

        for i, name in enumerate(model.output_fields):
            display = _label("outputs", name)
            tk.Label(self._out_frame, text=f"{display}:", font=FONT_LABEL,
                     anchor="e").grid(row=i, column=0, sticky="e",
                                      padx=(8, 10), pady=5)
            lbl = tk.Label(self._out_frame, text="--", font=FONT_BOLD, anchor="w")
            lbl.grid(row=i, column=1, sticky="w", padx=(0, 12), pady=5)
            self.output_labels[name] = lbl

    # ── prediction ────────────────────────────────────────────────────────────
    def _predict_start(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        self._predict_btn.config(state="disabled")
        self._pred_status.config(text="Running…")
        threading.Thread(target=self._predict_worker, daemon=True).start()

    def _predict_worker(self):
        try:
            import jax.numpy as jnp
            inputs = {}
            for name, ent in self.input_entries.items():
                raw = ent.get().strip()
                if not raw:
                    raise ValueError(f"Input '{name}' is empty.")
                inputs[name] = float(raw)

            x = jnp.array([inputs[k] for k in self.model.input_fields],
                           dtype=jnp.float32)
            y = self.model.predict_array(x).block_until_ready()
            y_np = np.asarray(y, dtype=float)
            self.root.after(0, lambda: self._update_ui(y_np))
        except Exception as exc:
            self.root.after(0, lambda: self._on_predict_error(str(exc)))

    def _update_ui(self, y_vals: np.ndarray):
        for name, lbl in self.output_labels.items():
            idx   = self.model.out_idx[name]
            scale = OUTPUT_SCALES.get(name, 1.0)
            val   = float(y_vals[idx]) * scale
            lbl.config(text=f"{val:.5g}")
        self._pred_status.config(text="")
        self._predict_btn.config(state="normal")

    def _on_predict_error(self, msg: str):
        self._pred_status.config(text="")
        self._predict_btn.config(state="normal")
        messagebox.showerror("Prediction Error", msg)

    # ── identifiability check ─────────────────────────────────────────────────
    def _open_identifiability(self):
        from gui_identifiability import open_identifiability_window
        open_identifiability_window(self.root, model=self.model)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    try:
        tmp = tk.Tk()
        tmp.destroy()
    except tk.TclError:
        pass

    root = tk.Tk()
    SurrogateGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
