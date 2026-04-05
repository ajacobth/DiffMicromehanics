"""gui.py – unified surrogate GUI for forward evaluation.

Supports the elastic, thermoelastic, and thermal conductivity surrogates.
Select the model, click Load, fill in the inputs, then click Predict.

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
import subprocess
import sys
import threading

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

import db.db as _db
from core.unit_manager import UM

# ── field label mapping ───────────────────────────────────────────────────────
_LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "field_labels.json")

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


# Output scales and unit labels are now provided dynamically by unit_manager.UM.
# See  data/units.json  to customise or add unit systems.

FONT_TITLE  = ("Helvetica", 17, "bold")
FONT_LABEL  = ("Helvetica", 15)
FONT_BOLD   = ("Helvetica", 15, "bold")
FONT_STATUS = ("Helvetica", 13, "italic")
FONT_ENTRY  = ("Helvetica", 15)

MODEL_NAMES = ["elastic", "thermoelastic", "thermal"]


# ── main GUI class ────────────────────────────────────────────────────────────
class SurrogateGUI:
    """Unified forward-evaluation GUI for elastic, thermoelastic, and thermal conductivity surrogates."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Composite Micromechanics Surrogate")
        root.geometry("1500x960")
        root.minsize(1100, 600)

        self.model:              object | None = None
        self.input_entries:      dict[str, tk.Entry] = {}
        self.input_prov_labels:  dict[str, tk.Label] = {}
        self.output_labels:      dict[str, tk.Label] = {}
        self._input_unit_labels: dict[str, tk.Label] = {}
        self._output_unit_labels: dict[str, tk.Label] = {}

        # material library state
        self._fiber_id:   int | None = None
        self._polymer_id: int | None = None
        self._fiber_map:  dict[str, int] = {}   # display name → id
        self._polymer_map: dict[str, int] = {}  # display name → id

        # last prediction — stored for "Save Prediction to Card"
        self._last_prediction_inputs:  dict[str, float] = {}
        self._last_prediction_outputs: dict[str, float] = {}
        self._save_pred_btn: tk.Widget | None = None

        self._build_layout()
        UM.register_callback(self._refresh_unit_labels)

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_layout(self):
        root = self.root
        root.grid_rowconfigure(4, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # ── top bar ──────────────────────────────────────────────────────────
        top = ttk.Frame(root, padding=(12, 10))
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(6, weight=1)

        tk.Label(top, text="Composite Surrogate Predictor",
                 font=FONT_TITLE).grid(row=0, column=0, sticky="w", padx=(0, 24))

        tk.Label(top, text="Model:", font=FONT_LABEL).grid(row=0, column=1, padx=(0, 6))

        self._model_var = tk.StringVar(value="elastic")
        for i, name in enumerate(MODEL_NAMES):
            ttk.Radiobutton(top, text=name.capitalize(), variable=self._model_var,
                            value=name).grid(row=0, column=2 + i, padx=6)

        self._load_btn = ttk.Button(top, text="Load Model", command=self._on_load)
        self._load_btn.grid(row=0, column=5, padx=(18, 6), sticky="w")

        self._load_status = tk.Label(top, text="No model loaded.",
                                     font=FONT_STATUS, fg="gray")
        self._load_status.grid(row=0, column=6, padx=10, sticky="w")

        ttk.Button(top, text="Identifiability Check",
                   command=self._open_identifiability).grid(
            row=0, column=7, padx=(12, 4), sticky="e"
        )

        ttk.Button(top, text="Material Card Viewer",
                   command=self._open_material_card_viewer).grid(
            row=0, column=8, padx=(4, 12), sticky="e"
        )

        ttk.Separator(top, orient="vertical").grid(row=0, column=9,
                                                    sticky="ns", padx=(6, 6))
        tk.Label(top, text="Units:", font=FONT_LABEL).grid(row=0, column=10, padx=(0, 4))
        self._unit_sys_var = tk.StringVar(value=UM.current_system)
        self._unit_sys_cb  = ttk.Combobox(top, textvariable=self._unit_sys_var,
                                           values=UM.available_systems,
                                           state="readonly", width=32,
                                           font=("Helvetica", 12))
        self._unit_sys_cb.grid(row=0, column=11, padx=4)
        self._unit_sys_cb.bind("<<ComboboxSelected>>", self._on_unit_system_change)

        ttk.Separator(root, orient="horizontal").grid(row=1, column=0, sticky="ew")

        # ── material library bar ──────────────────────────────────────────────
        mat = ttk.Frame(root, padding=(12, 6))
        mat.grid(row=2, column=0, sticky="ew")

        tk.Label(mat, text="Material Library:", font=FONT_BOLD).grid(
            row=0, column=0, padx=(0, 16), sticky="w")

        # fiber selector
        tk.Label(mat, text="Fiber:", font=FONT_LABEL).grid(
            row=0, column=1, padx=(0, 4))
        self._fiber_var = tk.StringVar(value="— none —")
        self._fiber_cb  = ttk.Combobox(
            mat, textvariable=self._fiber_var,
            state="disabled", width=28, font=FONT_LABEL)
        self._fiber_cb.grid(row=0, column=2, padx=(0, 16))
        self._fiber_cb.bind("<<ComboboxSelected>>", self._on_fiber_selected)

        # polymer selector
        tk.Label(mat, text="Polymer:", font=FONT_LABEL).grid(
            row=0, column=3, padx=(0, 4))
        self._polymer_var = tk.StringVar(value="— none —")
        self._polymer_cb  = ttk.Combobox(
            mat, textvariable=self._polymer_var,
            state="disabled", width=28, font=FONT_LABEL)
        self._polymer_cb.grid(row=0, column=4, padx=(0, 16))
        self._polymer_cb.bind("<<ComboboxSelected>>", self._on_polymer_selected)

        # source toggle (Neat / In-situ)
        self._src_var = tk.StringVar(value="neat")
        ttk.Radiobutton(mat, text="Neat",    variable=self._src_var,
                        value="neat",    command=self._on_source_change).grid(
            row=0, column=5, padx=4)
        self._insitu_rb = ttk.Radiobutton(
            mat, text="In-situ", variable=self._src_var,
            value="insitu", command=self._on_source_change, state="disabled")
        self._insitu_rb.grid(row=0, column=6, padx=(0, 20))

        # db status label
        self._db_status = tk.Label(
            mat, text="", font=FONT_STATUS, fg="gray")
        self._db_status.grid(row=0, column=7, sticky="w")

        ttk.Button(mat, text="Load from Card",
                   command=self._on_load_from_card).grid(
            row=0, column=8, padx=(16, 4))

        if not _db.db_exists():
            self._db_status.config(
                text="No database found — run init_db.py first", fg="red")
        else:
            self._populate_material_dropdowns()

        # Refresh dropdowns whenever the window regains focus (e.g. after
        # adding a material in the inverse GUI while this window is open).
        root.bind("<FocusIn>",
                  lambda e: self._populate_material_dropdowns()
                  if e.widget is root and _db.db_exists() else None)

        ttk.Separator(root, orient="horizontal").grid(row=3, column=0, sticky="ew")

        # ── main area: inputs (left) + outputs (right) ───────────────────────
        mid = ttk.Frame(root)
        mid.grid(row=4, column=0, sticky="nsew", padx=8, pady=6)
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
        self._save_pred_btn = ttk.Button(ctrl, text="Save Prediction to Card",
                                         command=self._on_save_prediction,
                                         state="disabled")
        self._save_pred_btn.pack(side="left", padx=4)

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
            from core.services import get_model, warm_up_model
            model = get_model(model_name)
            warm_up_model(model_name)
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
        # re-apply any already-selected materials now that input fields exist
        if _db.db_exists():
            self._fiber_cb.config(state="readonly")
            self._polymer_cb.config(state="readonly")
            if self._fiber_id is not None:
                self._apply_material_inputs()
            if self._polymer_id is not None:
                self._apply_material_inputs()

    def _on_load_error(self, msg: str):
        self._load_status.config(text="Load failed – see error.", fg="red")
        self._load_btn.config(state="normal")
        messagebox.showerror("Load Error", msg)

    # ── dynamic panel builders ────────────────────────────────────────────────
    def _rebuild_input_panel(self, model):
        for w in self._in_frame.winfo_children():
            w.destroy()
        self.input_entries       = {}
        self.input_prov_labels   = {}
        self._input_unit_labels  = {}
        self._in_frame.grid_columnconfigure(1, weight=1)

        for i, name in enumerate(model.input_fields):
            display = _label("inputs", name)
            tk.Label(self._in_frame, text=f"{display}:", font=FONT_LABEL,
                     anchor="e").grid(row=i, column=0, sticky="e",
                                      padx=(8, 10), pady=5)
            val_f = ttk.Frame(self._in_frame)
            val_f.grid(row=i, column=1, sticky="ew", padx=(0, 4), pady=5)
            ent = tk.Entry(val_f, width=18, font=FONT_ENTRY)
            ent.pack(side="left")
            ent.bind("<Return>", lambda _e: self._predict_start())
            ent.bind("<Key>", lambda _e, n=name: self._clear_prov(n))
            ulbl = tk.Label(val_f, text=UM.unit_label(name),
                            font=("Helvetica", 11), fg="gray", width=12, anchor="w")
            ulbl.pack(side="left", padx=(4, 0))
            self.input_entries[name]      = ent
            self._input_unit_labels[name] = ulbl
            prov_lbl = tk.Label(self._in_frame, text="", font=("Helvetica", 11),
                                fg="#888", anchor="w")
            prov_lbl.grid(row=i, column=2, sticky="w", padx=(0, 8), pady=5)
            self.input_prov_labels[name] = prov_lbl

    def _rebuild_output_panel(self, model):
        for w in self._out_frame.winfo_children():
            w.destroy()
        self.output_labels        = {}
        self._output_unit_labels  = {}
        self._out_frame.grid_columnconfigure(1, weight=1)

        for i, name in enumerate(model.output_fields):
            display = _label("outputs", name)
            tk.Label(self._out_frame, text=f"{display}:", font=FONT_LABEL,
                     anchor="e").grid(row=i, column=0, sticky="e",
                                      padx=(8, 10), pady=5)
            lbl = tk.Label(self._out_frame, text="--", font=FONT_BOLD, anchor="w")
            lbl.grid(row=i, column=1, sticky="w", padx=(0, 4), pady=5)
            ulbl = tk.Label(self._out_frame, text=UM.unit_label(name),
                            font=("Helvetica", 11), fg="gray", anchor="w")
            ulbl.grid(row=i, column=2, sticky="w", padx=(0, 8), pady=5)
            self.output_labels[name]       = lbl
            self._output_unit_labels[name] = ulbl

    # ── prediction ────────────────────────────────────────────────────────────
    def _predict_start(self):
        if self.model is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        missing = [
            _label("inputs", name)
            for name, ent in self.input_entries.items()
            if not ent.get().strip()
        ]
        if missing:
            messagebox.showerror(
                "Missing Inputs",
                "The following input(s) are missing:\n\n"
                + "\n".join(f"  \u2022 {m}" for m in missing),
            )
            return
        self._predict_btn.config(state="disabled")
        self._pred_status.config(text="Running\u2026")
        threading.Thread(target=self._predict_worker, daemon=True).start()

    # Orientation tensor field names (subset of possible input fields)
    _OT_FIELDS = ("a11", "a22", "a12", "a13", "a23")

    def _predict_worker(self):
        try:
            from core.services import run_forward, validate_orientation_tensor

            # ── 1. check for missing inputs ───────────────────────────────────
            missing = []
            for name, ent in self.input_entries.items():
                if not ent.get().strip():
                    display = _label("inputs", name)
                    missing.append(display)
            if missing:
                raise ValueError(
                    "The following input(s) are missing:\n\n"
                    + "\n".join(f"  • {m}" for m in missing)
                )

            # ── 2. parse inputs (convert display units → model units) ─────────
            inputs = {}
            for name, ent in self.input_entries.items():
                try:
                    inputs[name] = UM.from_display(name, float(ent.get().strip()))
                except ValueError:
                    display = _label("inputs", name)
                    raise ValueError(f"'{display}' is not a valid number.")

            # ── 3. orientation tensor PSD check ──────────────────────────────
            ot_present = all(f in inputs for f in self._OT_FIELDS)
            if ot_present:
                err = validate_orientation_tensor(inputs)
                if err:
                    raise ValueError(err)

            # ── 4. run prediction ─────────────────────────────────────────────
            outputs = run_forward(self._model_var.get(), inputs)
            self.root.after(0, lambda: self._update_ui(outputs))
        except Exception as exc:
            self.root.after(0, lambda exc=exc: self._on_predict_error(str(exc)))

    def _update_ui(self, outputs: dict):
        raw_outputs: dict[str, float] = {}
        for name, lbl in self.output_labels.items():
            raw = float(outputs[name])
            lbl.config(text=f"{UM.to_display(name, raw):.5g}")
            raw_outputs[name] = raw   # store in raw model units for saving

        # store for "Save Prediction to Card" — convert display → model units (MPa)
        self._last_prediction_inputs = {
            n: UM.from_display(n, float(ent.get().strip()))
            for n, ent in self.input_entries.items()
            if ent.get().strip()
        }
        self._last_prediction_outputs = raw_outputs

        self._pred_status.config(text="")
        self._predict_btn.config(state="normal")
        if self._save_pred_btn is not None:
            self._save_pred_btn.config(state="normal")

    def _on_predict_error(self, msg: str):
        self._pred_status.config(text="")
        self._predict_btn.config(state="normal")
        messagebox.showerror("Prediction Error", msg)

    # ── material library ──────────────────────────────────────────────────────
    def _populate_material_dropdowns(self):
        """Read fibers and polymers from DB and populate the comboboxes."""
        try:
            fibers   = _db.get_all_fibers()
            polymers = _db.get_all_polymers()
        except Exception as exc:
            self._db_status.config(text=f"DB error: {exc}", fg="red")
            return

        self._fiber_map   = {f["name"]: f["id"] for f in fibers}
        self._polymer_map = {p["name"]: p["id"] for p in polymers}

        fiber_names   = ["— none —"] + list(self._fiber_map)
        polymer_names = ["— none —"] + list(self._polymer_map)

        self._fiber_cb["values"]   = fiber_names
        self._polymer_cb["values"] = polymer_names
        self._fiber_cb.config(state="readonly")
        self._polymer_cb.config(state="readonly")

        n_f = len(fibers)
        n_p = len(polymers)
        self._db_status.config(
            text=f"Library: {n_f} fiber{'s' if n_f != 1 else ''}, "
                 f"{n_p} polymer{'s' if n_p != 1 else ''}",
            fg="gray")

    def _on_fiber_selected(self, _event=None):
        name = self._fiber_var.get()
        self._fiber_id = self._fiber_map.get(name)
        self._apply_material_inputs()
        self._refresh_insitu_toggle()
        self.root.focus_set()

    def _on_polymer_selected(self, _event=None):
        name = self._polymer_var.get()
        self._polymer_id = self._polymer_map.get(name)
        self._apply_material_inputs()
        self._refresh_insitu_toggle()
        self.root.focus_set()

    def _on_source_change(self):
        self._apply_material_inputs()

    def _apply_material_inputs(self):
        """Fill input fields from the selected fiber and/or polymer."""
        if not self.input_entries:
            return

        use_insitu = self._src_var.get() == "insitu"
        inputs: dict[str, float] = {}

        # ── fiber fields ──────────────────────────────────────────────────────
        if self._fiber_id is not None:
            if use_insitu:
                inputs.update(self._insitu_fiber_inputs() or
                              _db.fiber_model_inputs(self._fiber_id))
            else:
                inputs.update(_db.fiber_model_inputs(self._fiber_id))

        # ── polymer fields ────────────────────────────────────────────────────
        if self._polymer_id is not None:
            if use_insitu:
                inputs.update(self._insitu_polymer_inputs() or
                              _db.polymer_model_inputs(self._polymer_id))
            else:
                inputs.update(_db.polymer_model_inputs(self._polymer_id))

        # ── write into matching entry widgets (convert model units → display) ──
        for field, value in inputs.items():
            if field in self.input_entries and value is not None:
                ent = self.input_entries[field]
                ent.delete(0, tk.END)
                ent.insert(0, f"{UM.to_display(field, value):.6g}")

    def _refresh_insitu_toggle(self):
        """Enable the In-situ radio button only when inferred constituent data exists."""
        if self._fiber_id is None or self._polymer_id is None:
            self._insitu_rb.config(state="disabled")
            if self._src_var.get() == "insitu":
                self._src_var.set("neat")
            return
        try:
            f_props = _db.get_constituent_properties(
                "fiber", self._fiber_id, include_global=True)
            p_props = _db.get_constituent_properties(
                "polymer", self._polymer_id, include_global=True)
            has_insitu = any(
                p["source_tag"] == "inferred" for p in f_props + p_props
            )
        except Exception:
            has_insitu = False
        self._insitu_rb.config(state="normal" if has_insitu else "disabled")
        if not has_insitu and self._src_var.get() == "insitu":
            self._src_var.set("neat")

    def _insitu_fiber_inputs(self) -> dict | None:
        """Return most-recent inferred fiber constituent properties, keyed by model field name."""
        if self._fiber_id is None:
            return None
        try:
            props = _db.get_constituent_properties(
                "fiber", self._fiber_id, include_global=True)
        except Exception:
            return None
        seen: dict[str, float] = {}
        for p in props:
            if p["source_tag"] == "inferred" and p["property_name"] not in seen:
                seen[p["property_name"]] = float(p["value"])
        return seen if seen else None

    def _insitu_polymer_inputs(self) -> dict | None:
        """Return most-recent inferred polymer constituent properties, keyed by model field name."""
        if self._polymer_id is None:
            return None
        try:
            props = _db.get_constituent_properties(
                "polymer", self._polymer_id, include_global=True)
        except Exception:
            return None
        seen: dict[str, float] = {}
        for p in props:
            if p["source_tag"] == "inferred" and p["property_name"] not in seen:
                seen[p["property_name"]] = float(p["value"])
        return seen if seen else None

    # ── unit system ───────────────────────────────────────────────────────────

    def _on_unit_system_change(self, _event=None):
        new_sys = self._unit_sys_var.get()
        old_sys = UM.current_system
        if new_sys == old_sys:
            return
        # Convert existing values in input fields
        for name, ent in self.input_entries.items():
            raw = ent.get().strip()
            if not raw:
                continue
            try:
                new_val = UM.convert_between(name, float(raw), old_sys, new_sys)
                ent.delete(0, tk.END)
                ent.insert(0, f"{new_val:.6g}")
            except ValueError:
                pass
        UM.set_system(new_sys)   # fires UM callbacks → _refresh_unit_labels

    def _refresh_unit_labels(self):
        """Called by UM when any window changes the unit system."""
        if hasattr(self, "_unit_sys_var"):
            self._unit_sys_var.set(UM.current_system)
        for name, ulbl in self._input_unit_labels.items():
            ulbl.config(text=UM.unit_label(name))
        for name, ulbl in self._output_unit_labels.items():
            ulbl.config(text=UM.unit_label(name))
        # Re-render predicted output values in the new unit system
        for name, lbl in self.output_labels.items():
            if name in self._last_prediction_outputs:
                raw = self._last_prediction_outputs[name]
                lbl.config(text=f"{UM.to_display(name, raw):.5g}")

    # ── material card save / load / view ──────────────────────────────────────

    def _on_save_prediction(self):
        if not self._last_prediction_outputs:
            messagebox.showwarning("No prediction", "Run a prediction first.")
            return
        if not _db.db_exists():
            messagebox.showwarning("No database",
                                   "Run  python init_db.py  first.")
            return
        result = {
            "model":                 self._model_var.get() + "_forward",
            "free_variables":        {},
            "fixed_inputs":          self._last_prediction_inputs,
            "predicted_outputs":     self._last_prediction_outputs,
            "target_outputs":        {},
            "sigmas":                {},
            "final_optimiser_error": 0.0,
            "solver":                {},
        }
        from gui_card_dialogs import SaveToCardDialog
        SaveToCardDialog(self.root, result,
                         fiber_id=self._fiber_id,
                         polymer_id=self._polymer_id)

    def _on_load_from_card(self):
        if not _db.db_exists():
            messagebox.showwarning("No database",
                                   "Run  python init_db.py  first.")
            return
        from gui_card_dialogs import LoadFromCardDialog
        dlg = LoadFromCardDialog(self.root)
        if not dlg.loaded:
            return

        # Update fiber/polymer dropdowns to match the loaded card
        if dlg.loaded_card:
            fid = dlg.loaded_card.get("fiber_id")
            pid = dlg.loaded_card.get("polymer_id")
            # reverse lookup: id → display name
            fiber_name   = next((n for n, i in self._fiber_map.items()   if i == fid), None)
            polymer_name = next((n for n, i in self._polymer_map.items() if i == pid), None)
            if fiber_name:
                self._fiber_var.set(fiber_name)
                self._fiber_id = fid
            if polymer_name:
                self._polymer_var.set(polymer_name)
                self._polymer_id = pid
            self._refresh_insitu_toggle()

        for name, ent in self.input_entries.items():
            if name in dlg.loaded:
                ent.delete(0, tk.END)
                ent.insert(0, f"{UM.to_display(name, dlg.loaded[name]):.6g}")
                if name in self.input_prov_labels and name in dlg.loaded_provenance:
                    prov = dlg.loaded_provenance[name]
                    src  = prov.get("source_tag", "")
                    date = prov.get("date", "")
                    self.input_prov_labels[name].config(
                        text=f"[{src}, {date}]")

    def _open_material_card_viewer(self):
        try:
            from gui_material_card import MaterialCardViewer
            MaterialCardViewer(self.root)
        except Exception as exc:
            messagebox.showerror("Card Viewer Error", str(exc))

    def _clear_prov(self, field: str):
        lbl = self.input_prov_labels.get(field)
        if lbl:
            lbl.config(text="")

    # ── identifiability check ─────────────────────────────────────────────────
    def _open_identifiability(self):
        from gui_identifiability import open_identifiability_window
        open_identifiability_window(self.root, model=self.model,
                                    model_name=self._model_var.get())


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    SurrogateGUI(root)

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

    # Delay until after the event loop starts so the window is actually mapped.
    root.after(100, _bring_to_front)
    root.mainloop()


if __name__ == "__main__":
    main()
