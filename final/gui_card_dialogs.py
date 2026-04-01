"""gui_card_dialogs.py — shared material-card dialogs for gui_inverse.py and gui.py.

Two dialogs:
  SaveToCardDialog  — commit a solve/predict result to a material card
  LoadFromCardDialog — pull saved card data into inverse GUI input rows
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import db.db as _db

FONT_TITLE = ("Helvetica", 14, "bold")
FONT_BOLD  = ("Helvetica", 13, "bold")
FONT_LABEL = ("Helvetica", 13)
FONT_SMALL = ("Helvetica", 11)
FONT_ENTRY = ("Helvetica", 13)

# ── field classification ───────────────────────────────────────────────────────

# Surrogate input fields that belong to the microstructure snapshot
_MICRO_DB_MAP: dict[str, str] = {
    "a11": "a11", "a22": "a22", "a12": "a12", "a13": "a13", "a23": "a23",
    "ar":  "ar",  "ar_f": "ar",
    "fiber_massfrac": "mf", "w_f": "mf",
}
_MICRO_FIELDS = frozenset(_MICRO_DB_MAP)

# Fiber constituent input fields
_FIBER_FIELDS = frozenset({
    "e1", "e2", "g12", "f_nu12", "f_nu23", "fiber_density", "rho_f",
    "f_CTE1", "f_CTE2",
    "f_cte1", "f_cte2",   # thermoelastic model uses lowercase
    "k_f1", "k_f2",
})

# Polymer constituent input fields
_POLYMER_FIELDS = frozenset({
    "matrix_modulus", "matrix_poisson", "matrix_density", "rho_m",
    "matrix_CTE",
    "m_cte",              # thermoelastic model uses m_cte
    "k_m",
})

# Map from microstructure DB field names back to possible model input field names
# (used when loading from card into input rows)
_DB_TO_MODEL: dict[str, list[str]] = {
    "mf":  ["fiber_massfrac", "w_f"],
    "ar":  ["ar", "ar_f"],
    "a11": ["a11"], "a22": ["a22"], "a12": ["a12"],
    "a13": ["a13"], "a23": ["a23"],
}


# ══════════════════════════════════════════════════════════════════════════════
# SaveToCardDialog
# ══════════════════════════════════════════════════════════════════════════════

class SaveToCardDialog:
    """
    Modal dialog for saving a solve or prediction result to a material card.

    Parameters
    ----------
    parent : tk.Widget
    result : dict
        Must contain:
          "model"                  – surrogate model name string
          "free_variables"         – {field: value} dict (empty for forward runs)
          "fixed_inputs"           – {field: value} dict
          "predicted_outputs"      – {field: value} dict
          "target_outputs"         – {field: value} dict (can be empty)
          "sigmas"                 – {field: sigma} dict (can be empty)
          "final_optimiser_error"  – float loss
          "solver"                 – solver config dict
    """

    def __init__(self, parent: tk.Widget, result: dict,
                 fiber_id: Optional[int] = None,
                 polymer_id: Optional[int] = None):
        self.result     = result
        self.saved      = False
        self._preset_fiber_id   = fiber_id
        self._preset_polymer_id = polymer_id

        self._win = tk.Toplevel(parent)
        self._win.title("Save to Material Card")
        self._win.grab_set()
        self._win.resizable(True, True)
        self._win.geometry("580x720")
        self._win.minsize(560, 500)

        self._fiber_map:   dict[str, int] = {}
        self._polymer_map: dict[str, int] = {}
        self._printer_map: dict[str, int] = {}
        self._card_map:    dict[str, int] = {}  # display label → id

        self._build()
        self._load_db()
        self._win.wait_window()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self._win.grid_rowconfigure(0, weight=1)
        self._win.grid_columnconfigure(0, weight=1)

        # Scrollable canvas so the Save button stays visible regardless of card count
        _canvas = tk.Canvas(self._win, highlightthickness=0)
        _vsb = ttk.Scrollbar(self._win, orient="vertical", command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vsb.set)
        _vsb.grid(row=0, column=1, sticky="ns")
        _canvas.grid(row=0, column=0, sticky="nsew")

        outer = ttk.Frame(_canvas, padding=12)
        _cw = _canvas.create_window((0, 0), window=outer, anchor="nw")

        def _on_frame_resize(event):
            _canvas.configure(scrollregion=_canvas.bbox("all"))
        outer.bind("<Configure>", _on_frame_resize)

        def _on_canvas_resize(event):
            _canvas.itemconfig(_cw, width=event.width)
        _canvas.bind("<Configure>", _on_canvas_resize)

        def _on_mousewheel(event):
            _canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        _canvas.bind_all("<MouseWheel>", _on_mousewheel)

        outer.grid_columnconfigure(0, weight=1)

        tk.Label(outer, text="Save Result to Material Card",
                 font=FONT_TITLE).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Separator(outer, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=(0, 8))

        # ── material selectors ────────────────────────────────────────────────
        sel = ttk.LabelFrame(outer, text="Material + Printer", padding=8)
        sel.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        for row_idx, (lbl_text, var_attr, cb_attr) in enumerate([
            ("Fiber:",   "_fiber_var",   "_fiber_cb"),
            ("Polymer:", "_polymer_var", "_polymer_cb"),
            ("Printer:", "_printer_var", "_printer_cb"),
        ]):
            tk.Label(sel, text=lbl_text, font=FONT_LABEL,
                     width=10, anchor="e").grid(row=row_idx, column=0,
                                                 sticky="e", padx=(0, 8), pady=3)
            var = tk.StringVar(value="— select —")
            setattr(self, var_attr, var)
            cb = ttk.Combobox(sel, textvariable=var, state="readonly",
                              width=36, font=FONT_LABEL)
            setattr(self, cb_attr, cb)
            cb.grid(row=row_idx, column=1, pady=3, sticky="w")
            cb.bind("<<ComboboxSelected>>",
                    lambda _: [self._refresh_cards(), self._suggest_name()])

        ttk.Button(sel, text="+ Add printer…",
                   command=self._add_printer).grid(row=2, column=2, padx=(8, 0))

        # ── card selection ────────────────────────────────────────────────────
        card_lf = ttk.LabelFrame(outer, text="Card", padding=8)
        card_lf.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        card_lf.grid_columnconfigure(0, weight=1)

        self._card_var = tk.StringVar(value="new")
        self._cards_frame = ttk.Frame(card_lf)
        self._cards_frame.grid(row=0, column=0, sticky="ew")

        new_row = ttk.Frame(card_lf)
        new_row.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Radiobutton(new_row, text="Create new card:",
                        variable=self._card_var,
                        value="new").pack(side="left")
        self._card_name_var = tk.StringVar()
        tk.Entry(new_row, textvariable=self._card_name_var,
                 width=28, font=FONT_ENTRY).pack(side="left", padx=(8, 0))

        # ── processing conditions ─────────────────────────────────────────────
        pc_lf = ttk.LabelFrame(outer, text="Processing Conditions", padding=8)
        pc_lf.grid(row=4, column=0, sticky="ew", pady=(0, 8))

        _pc_fields = [
            ("Bead width (mm):",    "_pc_bead_width"),
            ("Bead height (mm):",   "_pc_bead_height"),
            ("Nozzle diam. (mm):",  "_pc_nozzle_diam"),
            ("Speed (mm/min):",     "_pc_speed"),
        ]
        for r_idx, (lbl_text, var_attr) in enumerate(_pc_fields):
            tk.Label(pc_lf, text=lbl_text, font=FONT_LABEL,
                     width=18, anchor="e").grid(row=r_idx, column=0,
                                                 sticky="e", padx=(0, 8), pady=2)
            var = tk.StringVar()
            setattr(self, var_attr, var)
            tk.Entry(pc_lf, textvariable=var, width=14,
                     font=FONT_ENTRY).grid(row=r_idx, column=1, sticky="w", pady=2)

        tk.Label(pc_lf, text="(Leave blank if not applicable)",
                 font=FONT_SMALL, fg="#888").grid(row=4, column=0, columnspan=2,
                                                   sticky="w", pady=(2, 0))

        # ── options ───────────────────────────────────────────────────────────
        opt_lf = ttk.LabelFrame(outer, text="Options", padding=8)
        opt_lf.grid(row=5, column=0, sticky="ew", pady=(0, 8))

        self._save_exp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_lf,
                        text="Save target values as experimental measurements",
                        variable=self._save_exp_var,
                        command=self._update_summary).pack(anchor="w", pady=2)

        tk.Label(opt_lf, text="Notes:", font=FONT_LABEL).pack(anchor="w", pady=(4, 0))
        self._notes_var = tk.StringVar()
        tk.Entry(opt_lf, textvariable=self._notes_var,
                 width=54, font=FONT_ENTRY).pack(fill="x", pady=(2, 0))

        # ── summary ───────────────────────────────────────────────────────────
        ttk.Separator(outer, orient="horizontal").grid(row=6, column=0,
                                                        sticky="ew", pady=(0, 4))
        self._summary_lbl = tk.Label(outer, text="", font=FONT_SMALL,
                                     fg="#555", justify="left", wraplength=520)
        self._summary_lbl.grid(row=7, column=0, sticky="w")
        self._update_summary()

        # ── buttons ───────────────────────────────────────────────────────────
        btn = ttk.Frame(outer)
        btn.grid(row=8, column=0, sticky="e", pady=(10, 0))
        ttk.Button(btn, text="Cancel",
                   command=self._win.destroy).pack(side="right", padx=(4, 0))
        ttk.Button(btn, text="Save",
                   command=self._on_save).pack(side="right", padx=(0, 4))

    # ── DB loading ────────────────────────────────────────────────────────────

    def _load_db(self):
        try:
            fibers   = _db.get_all_fibers()
            polymers = _db.get_all_polymers()
            printers = _db.get_all_printers()
        except Exception as exc:
            messagebox.showerror("DB Error", str(exc), parent=self._win)
            self._win.destroy()
            return

        self._fiber_map   = {f["name"]: f["id"] for f in fibers}
        self._polymer_map = {p["name"]: p["id"] for p in polymers}
        self._printer_map = {p["name"]: p["id"] for p in printers}

        self._fiber_cb["values"]   = list(self._fiber_map)
        self._polymer_cb["values"] = list(self._polymer_map)
        self._printer_cb["values"] = (
            list(self._printer_map) + ["— none (no specific printer) —"]
        )

        # pre-select fiber/polymer if IDs were passed in
        id_to_name = {v: k for k, v in self._fiber_map.items()}
        if self._preset_fiber_id is not None and self._preset_fiber_id in id_to_name:
            self._fiber_var.set(id_to_name[self._preset_fiber_id])

        id_to_name = {v: k for k, v in self._polymer_map.items()}
        if self._preset_polymer_id is not None and self._preset_polymer_id in id_to_name:
            self._polymer_var.set(id_to_name[self._preset_polymer_id])

        if self._preset_fiber_id is not None or self._preset_polymer_id is not None:
            self._refresh_cards()
            self._update_summary()

    def _refresh_cards(self):
        for w in self._cards_frame.winfo_children():
            w.destroy()
        self._card_map = {}

        fid = self._fiber_map.get(self._fiber_var.get())
        pid = self._polymer_map.get(self._polymer_var.get())
        if fid is None or pid is None:
            return

        try:
            all_cards = _db.get_all_print_configs()
            cards = [c for c in all_cards
                     if c["fiber_id"] == fid and c["polymer_id"] == pid]
        except Exception:
            return

        for c in cards:
            display = f"{c['name']}  (created {(c.get('created_at') or '')[:10]})"
            self._card_map[display] = c["id"]
            ttk.Radiobutton(self._cards_frame,
                            text=f"Append to existing:  {display}",
                            variable=self._card_var,
                            value=str(c["id"])).pack(anchor="w", pady=1)

    def _suggest_name(self):
        def _first(s: str) -> str:
            return s.split()[0] if s and s != "— select —" else ""
        parts = [
            _first(self._fiber_var.get()),
            _first(self._polymer_var.get()),
            _first(self._printer_var.get())
            if self._printer_var.get() not in ("— select —",
                                                "— none (no specific printer) —")
            else "",
        ]
        name = " / ".join(p for p in parts if p)
        if name:
            self._card_name_var.set(name)

    # ── add printer sub-dialog ────────────────────────────────────────────────

    def _add_printer(self):
        dlg = tk.Toplevel(self._win)
        dlg.title("Add Printer")
        dlg.grab_set()
        dlg.resizable(False, False)
        dlg.geometry("380x160")

        tk.Label(dlg, text="Name:", font=FONT_LABEL,
                 width=14, anchor="e").grid(row=0, column=0, padx=12, pady=8, sticky="e")
        name_var = tk.StringVar()
        tk.Entry(dlg, textvariable=name_var, width=24,
                 font=FONT_ENTRY).grid(row=0, column=1, padx=8, pady=8)

        tk.Label(dlg, text="Manufacturer:", font=FONT_LABEL,
                 width=14, anchor="e").grid(row=1, column=0, padx=12, pady=4, sticky="e")
        mfr_var = tk.StringVar()
        tk.Entry(dlg, textvariable=mfr_var, width=24,
                 font=FONT_ENTRY).grid(row=1, column=1, padx=8, pady=4)

        def _ok():
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Printer name is required.",
                                     parent=dlg)
                return
            try:
                _db.add_printer(name, manufacturer=mfr_var.get().strip())
                printers = _db.get_all_printers()
                self._printer_map = {p["name"]: p["id"] for p in printers}
                self._printer_cb["values"] = (
                    list(self._printer_map) + ["— none (no specific printer) —"]
                )
                self._printer_var.set(name)
            except Exception as exc:
                messagebox.showerror("DB Error", str(exc), parent=dlg)
            dlg.destroy()

        ttk.Button(dlg, text="Add", command=_ok).grid(
            row=2, column=1, padx=8, pady=8, sticky="e")
        dlg.wait_window()

    # ── summary label ─────────────────────────────────────────────────────────

    def _update_summary(self):
        free    = self.result.get("free_variables", {})
        outputs = self.result.get("predicted_outputs", {})
        targets = self.result.get("target_outputs", {})
        loss    = self.result.get("final_optimiser_error", 0.0)
        model   = self.result.get("model", "?")

        micro = [k for k in free if k in _MICRO_FIELDS]
        fiber = [k for k in free if k in _FIBER_FIELDS]
        poly  = [k for k in free if k in _POLYMER_FIELDS]

        lines = ["Will save:"]
        lines.append(
            f"  • inference_run  stage={model}  loss={loss:.3e}"
        )
        if micro:
            lines.append(
                f"  • microstructure_snapshot  [{', '.join(micro)} → inferred]"
            )
        else:
            # Forward run — microstructure fields are in fixed_inputs
            fixed = self.result.get("fixed_inputs", {})
            micro_f = [k for k in fixed if k in _MICRO_FIELDS]
            if micro_f:
                lines.append(
                    f"  • microstructure_snapshot  [{', '.join(micro_f)} → inputted]"
                )
        if fiber:
            lines.append(
                f"  • {len(fiber)} fiber constituent_property_values  [{', '.join(fiber)}]"
            )
        if poly:
            lines.append(
                f"  • {len(poly)} polymer constituent_property_values  [{', '.join(poly)}]"
            )
        lines.append(
            f"  • {len(outputs)} composite_property_values  [predicted]"
        )
        if self._save_exp_var.get() and targets:
            lines.append(
                f"  • {len(targets)} experimental_measurements  [target values + σ]"
            )
        self._summary_lbl.config(text="\n".join(lines))

    # ── save ──────────────────────────────────────────────────────────────────

    def _on_save(self):
        fid = self._fiber_map.get(self._fiber_var.get())
        pid = self._polymer_map.get(self._polymer_var.get())
        if fid is None or pid is None:
            messagebox.showerror("Missing selection",
                                 "Please select both a fiber and a polymer.",
                                 parent=self._win)
            return

        printer_name = self._printer_var.get()
        rid = self._printer_map.get(printer_name)  # None = no printer

        card_val = self._card_var.get()
        notes    = self._notes_var.get().strip()
        free     = self.result.get("free_variables", {})
        fixed    = self.result.get("fixed_inputs", {})
        all_in   = {**fixed, **free}
        outputs  = self.result.get("predicted_outputs", {})
        targets  = self.result.get("target_outputs", {})
        sigmas   = self.result.get("sigmas", {})
        model_nm = self.result.get("model", "unknown")
        loss     = float(self.result.get("final_optimiser_error", 0.0))
        solver   = self.result.get("solver", {})

        def _parse_float_opt(var: tk.StringVar) -> Optional[float]:
            s = var.get().strip()
            try:
                return float(s) if s else None
            except ValueError:
                return None

        try:
            # 1. Get or create print config ──────────────────────────────────
            if card_val == "new":
                card_name = (self._card_name_var.get().strip()
                             or f"{model_nm} card")

                # Create processing condition row (all optional)
                pc_id: Optional[int] = None
                bw  = _parse_float_opt(self._pc_bead_width)
                bh  = _parse_float_opt(self._pc_bead_height)
                nd  = _parse_float_opt(self._pc_nozzle_diam)
                spd = _parse_float_opt(self._pc_speed)
                if any(v is not None for v in (bw, bh, nd, spd)):
                    pc_id = _db.add_processing_condition(
                        bead_width=bw, bead_height=bh,
                        nozzle_diameter=nd, speed=spd,
                        notes=notes,
                    )

                cfg_id = _db.create_print_config(
                    name=card_name, fiber_id=fid, polymer_id=pid,
                    printer_id=rid, processing_condition_id=pc_id,
                    notes=notes)
            else:
                cfg_id = int(card_val)

            # 2. Build microstructure snapshot ────────────────────────────────
            micro_vals: dict[str, float | None] = {}
            micro_prov: dict[str, str]          = {}
            for src_field, db_field in _MICRO_DB_MAP.items():
                if src_field in all_in:
                    micro_vals[db_field] = all_in[src_field]
                    micro_prov[db_field] = (
                        "inferred" if src_field in free else "inputted"
                    )

            snap_id: Optional[int] = None
            if micro_vals:
                _SNAP_FIELDS = ("mf", "ar", "a11", "a22", "a12", "a13", "a23")
                latest = _db.get_latest_microstructure(cfg_id)
                changed = latest is None or any(
                    micro_vals.get(f) != latest.get(f) for f in _SNAP_FIELDS
                )
                if changed:
                    snap_id = _db.save_microstructure_snapshot(
                        print_config_id=cfg_id,
                        mf=micro_vals.get("mf"),
                        ar=micro_vals.get("ar"),
                        a11=micro_vals.get("a11"),
                        a22=micro_vals.get("a22"),
                        a12=micro_vals.get("a12"),
                        a13=micro_vals.get("a13"),
                        a23=micro_vals.get("a23"),
                        provenance=micro_prov,
                        notes=notes,
                    )
                else:
                    snap_id = latest["id"]

            # 3. Save inference run ────────────────────────────────────────────
            run_id = _db.save_inference_run(
                print_config_id=cfg_id,
                stage=model_nm,
                inputs=all_in,
                outputs=outputs,
                solver_cfg=solver,
                loss=loss,
                microstructure_snap_id=snap_id,
                notes=notes,
            )

            # 4. Constituent properties ───────────────────────────────────────
            # Free (inferred) variables — highest priority when loading back
            for field, value in free.items():
                if field in _FIBER_FIELDS:
                    _db.save_constituent_property(
                        constituent_type="fiber",
                        constituent_id=fid,
                        property_name=field,
                        value=value,
                        source_tag="inferred",
                        print_config_id=None,   # global
                        inference_run_id=run_id,
                        notes=notes,
                    )
                elif field in _POLYMER_FIELDS:
                    _db.save_constituent_property(
                        constituent_type="polymer",
                        constituent_id=pid,
                        property_name=field,
                        value=value,
                        source_tag="inferred",
                        print_config_id=None,   # global
                        inference_run_id=run_id,
                        notes=notes,
                    )
            # Fixed (inputted) fiber/polymer values — saved so forward-run
            # values can also be reloaded; inferred takes priority on load.
            # Only insert if the value has changed (or no row exists yet) to
            # avoid the table growing unboundedly on repeated forward saves.
            def _latest_by_tag(ctype, cid) -> tuple[dict[str, float], dict[str, float]]:
                """Return (inferred, inputted) dicts of most recent value per property."""
                inferred: dict[str, float] = {}
                inputted: dict[str, float] = {}
                for p in _db.get_constituent_properties(ctype, cid,
                                                        include_global=True):
                    name = p["property_name"]
                    tag  = p["source_tag"]
                    if tag == "inferred" and name not in inferred:
                        inferred[name] = float(p["value"])
                    elif tag == "inputted" and name not in inputted:
                        inputted[name] = float(p["value"])
                return inferred, inputted

            inf_f, inp_f = _latest_by_tag("fiber",   fid)
            inf_p, inp_p = _latest_by_tag("polymer", pid)

            for field, value in fixed.items():
                fval = float(value)
                if field in _FIBER_FIELDS:
                    # Never downgrade a previously-inferred property to "inputted",
                    # regardless of value (avoids float-precision mismatches between
                    # the stored inferred value and the re-parsed entry widget value).
                    if field in inf_f:
                        continue
                    if inp_f.get(field) == fval:
                        continue   # identical inputted value already stored
                    _db.save_constituent_property(
                        constituent_type="fiber",
                        constituent_id=fid,
                        property_name=field,
                        value=value,
                        source_tag="inputted",
                        print_config_id=None,
                        inference_run_id=run_id,
                        notes=notes,
                    )
                elif field in _POLYMER_FIELDS:
                    if field in inf_p:
                        continue
                    if inp_p.get(field) == fval:
                        continue
                    _db.save_constituent_property(
                        constituent_type="polymer",
                        constituent_id=pid,
                        property_name=field,
                        value=value,
                        source_tag="inputted",
                        print_config_id=None,
                        inference_run_id=run_id,
                        notes=notes,
                    )

            # 5. Composite properties (all predicted outputs) ─────────────────
            for prop, value in outputs.items():
                _db.save_composite_property(
                    print_config_id=cfg_id,
                    property_name=prop,
                    value=value,
                    source_tag="predicted",
                    inference_run_id=run_id,
                )

            # 6. Experimental measurements (optional) ─────────────────────────
            if self._save_exp_var.get():
                for prop, value in targets.items():
                    sigma = sigmas.get(prop, 0.0)
                    _db.save_experimental_measurement(
                        print_config_id=cfg_id,
                        property_name=prop,
                        value=value,
                        uncertainty=float(sigma) if sigma else None,
                        notes=notes,
                    )

            self.saved = True
            cfg_name   = (_db.get_print_config(cfg_id) or {}).get("name", str(cfg_id))
            messagebox.showinfo("Saved",
                                f"Saved to card '{cfg_name}'  (id={cfg_id})",
                                parent=self._win)
            self._win.destroy()

        except Exception as exc:
            messagebox.showerror("Save Error", str(exc), parent=self._win)


# ══════════════════════════════════════════════════════════════════════════════
# LoadFromCardDialog
# ══════════════════════════════════════════════════════════════════════════════

class LoadFromCardDialog:
    """
    Modal dialog for loading saved card data into the inverse GUI.

    After `wait_window`, check `.loaded`:
      None  – user cancelled
      dict  – {model_field_name: value} ready to paste into InputRow widgets
    """

    def __init__(self, parent: tk.Widget):
        self.loaded: Optional[dict] = None
        self.loaded_provenance: dict = {}
        self.loaded_card: Optional[dict] = None  # the print_config dict of the selected card

        self._win = tk.Toplevel(parent)
        self._win.title("Load from Material Card")
        self._win.grab_set()
        self._win.resizable(False, False)
        self._win.geometry("520x420")

        self._card_map: dict[str, dict] = {}  # display label → print_config dict
        self._build()
        self._load_db()
        self._win.wait_window()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        outer = ttk.Frame(self._win, padding=12)
        outer.pack(fill="both", expand=True)

        tk.Label(outer, text="Load Values from Material Card",
                 font=FONT_TITLE).pack(anchor="w", pady=(0, 6))
        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(0, 8))

        tk.Label(outer, text="Select card:", font=FONT_LABEL).pack(anchor="w")
        self._card_var = tk.StringVar()
        self._card_cb  = ttk.Combobox(outer, textvariable=self._card_var,
                                       state="readonly", width=54, font=FONT_LABEL)
        self._card_cb.pack(fill="x", pady=(2, 6))
        self._card_cb.bind("<<ComboboxSelected>>", lambda _: self._show_summary())

        self._summary_lbl = tk.Label(outer, text="", font=FONT_SMALL,
                                     fg="#555", justify="left", wraplength=480)
        self._summary_lbl.pack(anchor="w", pady=(0, 8))

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(0, 8))

        self._load_micro_var = tk.BooleanVar(value=True)
        self._load_const_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            outer,
            text="Load microstructure snapshot  (a11, a22, a12, a13, a23, ar, mf → Fixed)",
            variable=self._load_micro_var,
        ).pack(anchor="w", pady=2)
        ttk.Checkbutton(
            outer,
            text="Load inferred constituent properties  (matrix E, nu, CTE, k → Fixed)",
            variable=self._load_const_var,
        ).pack(anchor="w", pady=2)

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(8, 0))

        btn = ttk.Frame(outer)
        btn.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Button(btn, text="Cancel",
                   command=self._win.destroy).pack(side="right", padx=(4, 0))
        ttk.Button(btn, text="Load",
                   command=self._on_load).pack(side="right", padx=(0, 4))

    # ── DB loading ────────────────────────────────────────────────────────────

    def _load_db(self):
        try:
            cards = _db.get_all_print_configs()
        except Exception as exc:
            messagebox.showerror("DB Error", str(exc), parent=self._win)
            return

        for c in cards:
            f = _db.get_fiber(c["fiber_id"])
            p = _db.get_polymer(c["polymer_id"])
            f_name = f["name"] if f else f"fiber:{c['fiber_id']}"
            p_name = p["name"] if p else f"polymer:{c['polymer_id']}"
            display = (
                f"{c['name']}  "
                f"[{f_name} / {p_name}]  "
                f"(id={c['id']})"
            )
            self._card_map[display] = c

        self._card_cb["values"] = list(self._card_map)

    def _show_summary(self):
        card = self._card_map.get(self._card_var.get())
        if not card:
            return

        cfg_id = card["id"]
        fid    = card["fiber_id"]
        pid    = card["polymer_id"]
        lines: list[str] = []

        try:
            snap = _db.get_latest_microstructure(cfg_id)
            if snap:
                lines.append(
                    f"Latest microstructure: "
                    f"mf={snap.get('mf')}, ar={snap.get('ar')}, "
                    f"a11={snap.get('a11')}, a22={snap.get('a22')}"
                )
            else:
                lines.append("No microstructure snapshot found.")

            f_props = _db.get_constituent_properties("fiber",   fid, include_global=True)
            p_props = _db.get_constituent_properties("polymer", pid, include_global=True)

            inf_f = [p for p in f_props   if p["source_tag"] == "inferred"]
            inf_p = [p for p in p_props   if p["source_tag"] == "inferred"]

            if inf_f:
                names = ", ".join(p["property_name"] for p in inf_f[:5])
                lines.append(f"Fiber inferred props: {names}")
            else:
                lines.append("No inferred fiber properties.")

            if inf_p:
                names = ", ".join(p["property_name"] for p in inf_p[:5])
                lines.append(f"Polymer inferred props: {names}")
            else:
                lines.append("No inferred polymer properties.")

        except Exception:
            lines.append("Could not load card details.")

        self._summary_lbl.config(text="\n".join(lines))

    # ── load ──────────────────────────────────────────────────────────────────

    def _on_load(self):
        card = self._card_map.get(self._card_var.get())
        if not card:
            messagebox.showerror("No card selected",
                                 "Please select a card.", parent=self._win)
            return

        cfg_id = card["id"]
        fid    = card["fiber_id"]
        pid    = card["polymer_id"]
        loaded: dict[str, float] = {}
        loaded_provenance: dict[str, dict] = {}

        try:
            if self._load_micro_var.get():
                snap = _db.get_latest_microstructure(cfg_id)
                if snap:
                    prov = snap.get("provenance") or {}
                    date = (snap.get("created_at") or "")[:10]
                    for db_field, model_fields in _DB_TO_MODEL.items():
                        val = snap.get(db_field)
                        if val is not None:
                            src = prov.get(db_field) or "inputted"
                            for mf in model_fields:
                                loaded[mf] = float(val)
                                loaded_provenance[mf] = {
                                    "source_tag": src,
                                    "date": date,
                                }

            if self._load_const_var.get():
                f_props = _db.get_constituent_properties(
                    "fiber", fid,
                    print_config_id=cfg_id, include_global=True,
                )
                p_props = _db.get_constituent_properties(
                    "polymer", pid,
                    print_config_id=cfg_id, include_global=True,
                )

                # Two-pass: inferred always wins over inputted regardless of
                # which was saved more recently.  Within the same tag, the DB
                # returns newest-first so the first occurrence is most recent.
                for props in (f_props, p_props):
                    inferred_best: dict[str, dict] = {}
                    inputted_best: dict[str, dict] = {}
                    for p in props:
                        name = p["property_name"]
                        if p["source_tag"] == "inferred" and name not in inferred_best:
                            inferred_best[name] = p
                        elif p["source_tag"] == "inputted" and name not in inputted_best:
                            inputted_best[name] = p

                    for name, p in inferred_best.items():
                        loaded[name] = float(p["value"])
                        loaded_provenance[name] = {
                            "source_tag": "inferred",
                            "date": (p.get("created_at") or "")[:10],
                        }
                    for name, p in inputted_best.items():
                        if name not in loaded:   # don't overwrite inferred
                            loaded[name] = float(p["value"])
                            loaded_provenance[name] = {
                                "source_tag": "inputted",
                                "date": (p.get("created_at") or "")[:10],
                            }

        except Exception as exc:
            messagebox.showerror("Load Error", str(exc), parent=self._win)
            return

        # Fallback: fill any still-missing constituent fields from the
        # neat (datasheet) values stored on the fiber/polymer record.
        # This ensures old cards — saved before constituent properties were
        # written to the DB — still populate the input fields.
        _FIBER_NEAT_MAP = {
            "e1": "neat_E1", "e2": "neat_E2", "g12": "neat_G12",
            "f_nu12": "neat_nu12", "f_nu23": "neat_nu23",
            "fiber_density": "neat_rho",
            "k_f1": "neat_k1", "k_f2": "neat_k2",
            "f_CTE1": "neat_CTE1", "f_CTE2": "neat_CTE2",
            "f_cte1": "neat_CTE1", "f_cte2": "neat_CTE2",  # thermoelastic model names
        }
        _POLY_NEAT_MAP = {
            "matrix_modulus": "neat_E1", "matrix_poisson": "neat_nu12",
            "matrix_density": "neat_rho", "k_m": "neat_k",
            "matrix_CTE": "neat_CTE",
            "m_cte": "neat_CTE",  # thermoelastic model name
        }
        try:
            fiber_rec = _db.get_fiber(fid)
            poly_rec  = _db.get_polymer(pid)
            if fiber_rec:
                for mf, nf in _FIBER_NEAT_MAP.items():
                    if mf not in loaded:
                        val = fiber_rec.get(nf)
                        if val is not None:
                            loaded[mf] = float(val)
                            loaded_provenance[mf] = {"source_tag": "neat", "date": "—"}
            if poly_rec:
                for mf, nf in _POLY_NEAT_MAP.items():
                    if mf not in loaded:
                        val = poly_rec.get(nf)
                        if val is not None:
                            loaded[mf] = float(val)
                            loaded_provenance[mf] = {"source_tag": "neat", "date": "—"}
        except Exception:
            pass  # neat fallback is best-effort

        if not loaded:
            messagebox.showwarning("Nothing to load",
                                   "No values found for this card.",
                                   parent=self._win)
            return

        self.loaded = loaded
        self.loaded_provenance = loaded_provenance
        self.loaded_card = card
        self._win.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# SaveThermalToCardDialog
# ══════════════════════════════════════════════════════════════════════════════

class SaveThermalToCardDialog:
    """
    Modal dialog for saving thermal inverse results to a material card.

    Parameters
    ----------
    parent : tk.Widget
    params : object
        best_params with attributes p1, p2, l2, t
    fixed_inputs : dict
        Fixed structural inputs used in the run (ar_f, w_f, a11, a22, ...)
    temperatures : np.ndarray
    K_pred : np.ndarray  shape (n_temps, 3)  — K11, K22, K33 predictions
    loss : float
    """

    def __init__(self, parent, params, fixed_inputs: dict,
                 temperatures, K_pred, loss: float):
        self._params       = params
        self._fixed_inputs = fixed_inputs
        self._temperatures = temperatures
        self._K_pred       = K_pred
        self._loss         = loss
        self.saved         = False

        self._win = tk.Toplevel(parent)
        self._win.title("Save Thermal Results to Card")
        self._win.grab_set()
        self._win.resizable(False, False)
        self._win.geometry("520x480")

        self._fiber_map   = {}
        self._polymer_map = {}
        self._printer_map = {}
        self._card_map    = {}

        self._build()
        self._load_db()
        self._win.wait_window()

    def _build(self):
        self._win.grid_rowconfigure(0, weight=1)
        self._win.grid_columnconfigure(0, weight=1)

        outer = ttk.Frame(self._win, padding=12)
        outer.grid(sticky="nsew")
        outer.grid_columnconfigure(0, weight=1)

        tk.Label(outer, text="Save Thermal Inverse Results to Card",
                 font=FONT_TITLE).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Separator(outer, orient="horizontal").grid(
            row=1, column=0, sticky="ew", pady=(0, 8))

        sel = ttk.LabelFrame(outer, text="Material + Printer", padding=8)
        sel.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        for row_idx, (lbl_text, var_attr, cb_attr) in enumerate([
            ("Fiber:",   "_fiber_var",   "_fiber_cb"),
            ("Polymer:", "_polymer_var", "_polymer_cb"),
            ("Printer:", "_printer_var", "_printer_cb"),
        ]):
            tk.Label(sel, text=lbl_text, font=FONT_LABEL,
                     width=10, anchor="e").grid(
                row=row_idx, column=0, sticky="e", padx=(0, 8), pady=3)
            var = tk.StringVar(value="— select —")
            setattr(self, var_attr, var)
            cb = ttk.Combobox(sel, textvariable=var, state="readonly",
                              width=34, font=FONT_LABEL)
            setattr(self, cb_attr, cb)
            cb.grid(row=row_idx, column=1, pady=3, sticky="w")
            cb.bind("<<ComboboxSelected>>",
                    lambda _: self._refresh_cards())

        card_lf = ttk.LabelFrame(outer, text="Card", padding=8)
        card_lf.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        card_lf.grid_columnconfigure(0, weight=1)

        self._card_var    = tk.StringVar(value="new")
        self._cards_frame = ttk.Frame(card_lf)
        self._cards_frame.grid(row=0, column=0, sticky="ew")

        new_row = ttk.Frame(card_lf)
        new_row.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Radiobutton(new_row, text="Create new card:",
                        variable=self._card_var,
                        value="new").pack(side="left")
        self._card_name_var = tk.StringVar()
        tk.Entry(new_row, textvariable=self._card_name_var,
                 width=26, font=FONT_ENTRY).pack(side="left", padx=(8, 0))

        # summary
        summary_lines = [
            f"Will save:  p1={self._params.p1:.4g}  p2={self._params.p2:.4g}"
            f"  l2={self._params.l2:.4g}  t={self._params.t:.4g}",
            f"  loss={self._loss:.4e}   temps={len(self._temperatures)}"
            f"   K predictions: {self._K_pred.shape[0]} × 3",
        ]
        tk.Label(outer, text="\n".join(summary_lines),
                 font=FONT_SMALL, fg="#555",
                 justify="left").grid(row=4, column=0, sticky="w", pady=(0, 8))

        btn = ttk.Frame(outer)
        btn.grid(row=5, column=0, sticky="e", pady=(10, 0))
        ttk.Button(btn, text="Cancel",
                   command=self._win.destroy).pack(side="right", padx=(4, 0))
        ttk.Button(btn, text="Save",
                   command=self._on_save).pack(side="right", padx=(0, 4))

    def _load_db(self):
        try:
            fibers   = _db.get_all_fibers()
            polymers = _db.get_all_polymers()
            printers = _db.get_all_printers()
        except Exception as exc:
            messagebox.showerror("DB Error", str(exc), parent=self._win)
            self._win.destroy()
            return
        self._fiber_map   = {f["name"]: f["id"] for f in fibers}
        self._polymer_map = {p["name"]: p["id"] for p in polymers}
        self._printer_map = {p["name"]: p["id"] for p in printers}
        self._fiber_cb["values"]   = list(self._fiber_map)
        self._polymer_cb["values"] = list(self._polymer_map)
        self._printer_cb["values"] = (
            list(self._printer_map) + ["— none —"])

    def _refresh_cards(self):
        for w in self._cards_frame.winfo_children():
            w.destroy()
        self._card_map = {}
        fid = self._fiber_map.get(self._fiber_var.get())
        pid = self._polymer_map.get(self._polymer_var.get())
        if fid is None or pid is None:
            return
        try:
            all_cards = _db.get_all_print_configs()
            cards = [c for c in all_cards
                     if c["fiber_id"] == fid and c["polymer_id"] == pid]
        except Exception:
            return
        for c in cards:
            display = f"{c['name']}  (created {(c.get('created_at') or '')[:10]})"
            self._card_map[display] = c["id"]
            ttk.Radiobutton(self._cards_frame,
                            text=f"Append to existing:  {display}",
                            variable=self._card_var,
                            value=str(c["id"])).pack(anchor="w", pady=1)

    def _on_save(self):
        import numpy as np
        fid = self._fiber_map.get(self._fiber_var.get())
        pid = self._polymer_map.get(self._polymer_var.get())
        if fid is None or pid is None:
            messagebox.showerror("Missing selection",
                                 "Please select both a fiber and a polymer.",
                                 parent=self._win)
            return

        printer_name = self._printer_var.get()
        rid = self._printer_map.get(printer_name)

        card_val = self._card_var.get()
        try:
            if card_val == "new":
                card_name = (self._card_name_var.get().strip()
                             or "thermal card")
                cfg_id = _db.create_print_config(
                    name=card_name, fiber_id=fid, polymer_id=pid,
                    printer_id=rid)
            else:
                cfg_id = int(card_val)

            # microstructure snapshot from fixed inputs — only insert if changed
            fi = self._fixed_inputs
            micro_prov = {k: "inputted" for k in ("mf", "ar", "a11", "a22",
                                                    "a12", "a13", "a23")}
            _thermal_micro = {
                "mf": fi.get("w_f"), "ar": fi.get("ar_f"),
                "a11": fi.get("a11"), "a22": fi.get("a22"),
                "a12": fi.get("a12"), "a13": fi.get("a13"), "a23": fi.get("a23"),
            }
            _SNAP_FIELDS = ("mf", "ar", "a11", "a22", "a12", "a13", "a23")
            _t_latest = _db.get_latest_microstructure(cfg_id)
            _t_changed = _t_latest is None or any(
                _thermal_micro.get(f) != _t_latest.get(f) for f in _SNAP_FIELDS
            )
            if _t_changed:
                snap_id = _db.save_microstructure_snapshot(
                    print_config_id=cfg_id,
                    mf=_thermal_micro["mf"],
                    ar=_thermal_micro["ar"],
                    a11=_thermal_micro["a11"],
                    a22=_thermal_micro["a22"],
                    a12=_thermal_micro["a12"],
                    a13=_thermal_micro["a13"],
                    a23=_thermal_micro["a23"],
                    provenance=micro_prov,
                )
            else:
                snap_id = _t_latest["id"]

            # thermal inverse results (constituent properties)
            p = self._params
            run_id = _db.save_thermal_inverse_results(
                print_config_id=cfg_id,
                fiber_id=fid,
                polymer_id=pid,
                parametric_outputs={
                    "p1": float(p.p1), "p2": float(p.p2),
                    "l2": float(p.l2), "t":  float(p.t),
                },
                solver_cfg={},
                loss=float(self._loss),
                microstructure_snap_id=snap_id,
            )

            # composite K predictions at each temperature
            for i, temp in enumerate(self._temperatures):
                for col, prop in enumerate(("K11", "K22", "K33")):
                    val = float(self._K_pred[i, col])
                    _db.save_composite_property(
                        print_config_id=cfg_id,
                        property_name=prop,
                        value=val,
                        unit="W/m·K",
                        source_tag="predicted",
                        inference_run_id=run_id,
                        temperature_C=float(temp),
                    )

            self.saved = True
            cfg_name = (_db.get_print_config(cfg_id) or {}).get("name", str(cfg_id))
            messagebox.showinfo("Saved",
                                f"Saved to card '{cfg_name}'  (id={cfg_id})",
                                parent=self._win)
            self._win.destroy()

        except Exception as exc:
            messagebox.showerror("Save Error", str(exc), parent=self._win)
