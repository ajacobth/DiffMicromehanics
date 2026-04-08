"""gui_transfer.py — Transfer a material card to a new printer.

Opens from gui.py via "Transfer to New Printer…" button.

Workflow:
  ① Pick a source card  →  shows constituent props (locked, from DB)
  ② Enter microstructure for the new printer  (manual  OR  infer from measurements)
  ③ Forward-predict one or more models, then Save to a new card

No new DB write logic — saving delegates entirely to SaveToCardDialog.
"""
from __future__ import annotations

# JAX backend must be pinned to CPU before the first `import jax`.
# Metal (Apple GPU) only supports float32; the inverse solver requires float64.
import os as _os
_os.environ.setdefault("JAX_ENABLE_X64",    "1")
_os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import threading
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

import db.db as _db
from core.unit_manager import UM

FONT_TITLE  = ("Helvetica", 15, "bold")
FONT_BOLD   = ("Helvetica", 13, "bold")
FONT_LABEL  = ("Helvetica", 13)
FONT_SMALL  = ("Helvetica", 11)
FONT_STATUS = ("Helvetica", 12, "italic")
FONT_ENTRY  = ("Helvetica", 13)

# ── Microstructure fields ──────────────────────────────────────────────────────

_MICRO_FIELDS = ("a11", "a22", "a12", "a13", "a23", "ar", "fiber_massfrac")
_MICRO_LABELS = {
    "a11": "a₁₁",
    "a22": "a₂₂",
    "a12": "a₁₂",
    "a13": "a₁₃",
    "a23": "a₂₃",
    "ar":  "Aspect ratio",
    "fiber_massfrac": "Mass fraction",
}
_MICRO_BOUNDS = {
    "a11":            (0.0,  1.0),
    "a22":            (0.0,  1.0),
    "a12":            (-0.5, 0.5),
    "a13":            (-0.5, 0.5),
    "a23":            (-0.5, 0.5),
    "ar":             (1.0, 100.0),
    "fiber_massfrac": (0.01, 0.60),
}

# ── Constituent property metadata ─────────────────────────────────────────────

_CONST_LABELS = {
    "matrix_modulus": "Matrix modulus",
    "matrix_poisson": "Matrix Poisson ν",
    "f_cte1":         "Fiber CTE α₁₁",
    "f_cte2":         "Fiber CTE α₂₂",
    "m_cte":          "Matrix CTE",
    "k_f1":           "Fiber k‖",
    "k_f2":           "Fiber k⊥",
    "k_m":            "Matrix k",
}
_CONST_UNITS = {
    "matrix_modulus": "MPa",
    "matrix_poisson": "",
    "f_cte1":         "1/K",
    "f_cte2":         "1/K",
    "m_cte":          "1/K",
    "k_f1":           "W/m·K",
    "k_f2":           "W/m·K",
    "k_m":            "W/m·K",
}
_CONST_STAGE = {
    "matrix_modulus": "Stage 1",
    "matrix_poisson": "Stage 1",
    "f_cte1":         "Stage 2",
    "f_cte2":         "Stage 2",
    "m_cte":          "Stage 2",
    "k_f1":           "Stage 3",
    "k_f2":           "Stage 3",
    "k_m":            "Stage 3",
}

# Elastic outputs available as transfer inverse targets
# (all 9 outputs of the elastic surrogate)
_TARGET_FIELDS = [
    ("E1",   "E₁"),
    ("E2",   "E₂"),
    ("E3",   "E₃"),
    ("G12",  "G₁₂"),
    ("G13",  "G₁₃"),
    ("G23",  "G₂₃"),
    ("nu12", "ν₁₂"),
    ("nu13", "ν₁₃"),
    ("nu23", "ν₂₃"),
]


# ── Microstructure input row (for infer pane) ────────────────────────────────

class MicroRow:
    """One microstructure input row in the infer pane.

    Shows Fixed / Free radio buttons.  When Free → bounds entries are revealed.
    The value entry is always visible and doubles as the initial guess.
    """

    def __init__(self, parent: tk.Widget, row: int, field: str, label: str,
                 default_bounds: tuple[float, float]):
        self.field = field
        self._default_lo, self._default_hi = default_bounds
        self._mode = tk.StringVar(value="free")

        # label
        tk.Label(parent, text=f"{label}:", font=FONT_LABEL, anchor="e",
                 width=17).grid(row=row, column=0, sticky="e", padx=(4, 6), pady=2)

        # Fixed / Free radios
        rb_frame = ttk.Frame(parent)
        rb_frame.grid(row=row, column=1, sticky="w", padx=2)
        ttk.Radiobutton(rb_frame, text="Fixed", variable=self._mode,
                        value="fixed", command=self._on_mode).pack(side="left")
        ttk.Radiobutton(rb_frame, text="Free",  variable=self._mode,
                        value="free",  command=self._on_mode).pack(side="left", padx=(8, 0))

        # value / initial-guess entry
        vf = ttk.Frame(parent)
        vf.grid(row=row, column=2, padx=4, pady=2)
        self._val = tk.Entry(vf, width=10, font=FONT_ENTRY)
        self._val.pack(side="left")

        # bounds frame — shown when Free
        self._bf = ttk.Frame(parent)
        self._bf.grid(row=row, column=3, sticky="w", padx=2)
        tk.Label(self._bf, text="lo:", font=FONT_SMALL).pack(side="left")
        self._lo = tk.Entry(self._bf, width=7, font=FONT_ENTRY)
        self._lo.pack(side="left", padx=(2, 6))
        self._lo.insert(0, str(self._default_lo))
        tk.Label(self._bf, text="hi:", font=FONT_SMALL).pack(side="left")
        self._hi = tk.Entry(self._bf, width=7, font=FONT_ENTRY)
        self._hi.pack(side="left", padx=(2, 0))
        self._hi.insert(0, str(self._default_hi))
        tk.Label(self._bf, text="(leave blank for defaults)",
                 font=FONT_SMALL, fg="gray").pack(side="left", padx=(6, 0))

        # start in free mode (bounds visible)
        self._on_mode()

    def _on_mode(self):
        if self._mode.get() == "free":
            self._bf.grid()
        else:
            self._bf.grid_remove()

    def is_free(self) -> bool:
        return self._mode.get() == "free"

    def get_value(self) -> Optional[float]:
        raw = self._val.get().strip()
        if not raw:
            return None
        return float(raw)

    def get_bounds(self) -> tuple[float, float]:
        lo = self._lo.get().strip()
        hi = self._hi.get().strip()
        if lo and hi:
            return (float(lo), float(hi))
        return (self._default_lo, self._default_hi)

    def set_value(self, v: float):
        self._val.delete(0, tk.END)
        self._val.insert(0, f"{v:.5g}")




# ══════════════════════════════════════════════════════════════════════════════
class TransferWindow:
    """Standalone top-level window for cross-printer material transfer."""

    def __init__(self, parent: tk.Widget):
        self._parent = parent
        self._win = tk.Toplevel(parent)
        self._win.title("Transfer Material to New Printer")
        self._win.geometry("1380x740")
        self._win.minsize(1100, 600)
        self._win.resizable(True, True)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Explicit widget references used by unlock helpers
        self._mode_rbs: list = []   # manual / infer radio buttons
        self._model_rbs: list = []  # elastic / thermoelastic / thermal radio buttons

        # ── state ──────────────────────────────────────────────────────────
        self._source_card_id:   Optional[int]  = None
        self._source_fid:       Optional[int]  = None
        self._source_pid:       Optional[int]  = None
        self._constituent_props: dict           = {}   # key → {value, source_tag} | None
        self._micro_inferred:   bool            = False  # True if ② was solved by inverse
        self._last_inputs:      dict[str, float] = {}
        self._last_outputs:     dict[str, float] = {}
        self._last_model_name:  str              = ""
        self._card_map:         dict[str, dict]  = {}   # display label → {id, fiber_id, polymer_id}

        # microstructure entry widgets (manual pane)
        self._micro_entries: dict[str, tk.Entry] = {}

        # infer-mode microstructure rows: field → MicroRow
        self._micro_rows: dict[str, MicroRow] = {}

        # infer-mode target widgets: field → (active BooleanVar, entry, sigma_entry)
        self._target_vars: dict[str, tuple[tk.BooleanVar, tk.Entry, tk.Entry]] = {}

        self._build()
        self._load_cards()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self._win.grid_rowconfigure(2, weight=1)
        self._win.grid_columnconfigure(0, weight=1)

        # ── top bar: title + card selector ───────────────────────────────────
        top = ttk.Frame(self._win, padding=(12, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(2, weight=1)

        tk.Label(top, text="Transfer Material to New Printer",
                 font=FONT_TITLE).grid(row=0, column=0, sticky="w", padx=(0, 24))
        tk.Label(top, text="Source card:", font=FONT_LABEL).grid(
            row=0, column=1, padx=(0, 6))

        self._card_var = tk.StringVar(value="— select —")
        self._card_cb  = ttk.Combobox(top, textvariable=self._card_var,
                                      state="readonly", font=FONT_LABEL, width=52)
        self._card_cb.grid(row=0, column=2, sticky="w")
        self._card_cb.bind("<<ComboboxSelected>>", self._on_card_selected)

        ttk.Separator(self._win, orient="horizontal").grid(
            row=1, column=0, sticky="ew")

        # ── main 3-column area ────────────────────────────────────────────────
        main = ttk.Frame(self._win)
        main.grid(row=2, column=0, sticky="nsew", padx=8, pady=6)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=3, minsize=320)
        main.grid_columnconfigure(1, weight=0)
        main.grid_columnconfigure(2, weight=3, minsize=380)
        main.grid_columnconfigure(3, weight=0)
        main.grid_columnconfigure(4, weight=2, minsize=300)

        left  = ttk.LabelFrame(main, text="① Constituent Properties  (from source card)", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ttk.Separator(main, orient="vertical").grid(row=0, column=1, sticky="ns", padx=4)

        mid   = ttk.LabelFrame(main, text="② Microstructure for New Printer  (locked until card selected)", padding=10)
        mid.grid(row=0, column=2, sticky="nsew", padx=4)
        mid.grid_rowconfigure(1, weight=1)
        mid.grid_columnconfigure(0, weight=1)

        ttk.Separator(main, orient="vertical").grid(row=0, column=3, sticky="ns", padx=4)

        right = ttk.LabelFrame(main, text="③ Predict & Save  (locked until card selected)", padding=10)
        right.grid(row=0, column=4, sticky="nsew", padx=(4, 0))
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._sec2 = mid
        self._sec3 = right

        self._build_section1(left)
        self._build_section2(mid)
        self._build_section3(right)

    # ── Section ①: Constituent properties (left panel) ───────────────────────

    def _build_section1(self, panel: ttk.LabelFrame):
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)

        self._const_frame = ttk.Frame(panel)
        self._const_frame.grid(row=0, column=0, sticky="nsew")
        tk.Label(self._const_frame,
                 text="Select a card above to see constituent properties.",
                 font=FONT_STATUS, fg="gray").pack(anchor="w", pady=10)

    def _load_cards(self):
        if not _db.db_exists():
            return
        try:
            from core.services.service_material import (
                list_cards, list_fibers, list_polymers, list_printers,
            )
            cards    = list_cards()
            fibers   = {f["id"]: f["name"] for f in list_fibers()}
            polymers = {p["id"]: p["name"] for p in list_polymers()}
            printers = {p["id"]: p["name"] for p in list_printers()}
        except Exception as exc:
            messagebox.showerror("DB Error", str(exc), parent=self._win)
            return

        self._card_map = {}
        for c in cards:
            fname = fibers.get(c["fiber_id"], "?")
            pname = polymers.get(c["polymer_id"], "?")
            rname = printers.get(c.get("printer_id"), "—")
            label = f"{c['name']}   ({fname} / {pname} / {rname})"
            self._card_map[label] = {
                "id":         c["id"],
                "fiber_id":   c["fiber_id"],
                "polymer_id": c["polymer_id"],
            }

        self._card_cb["values"] = list(self._card_map)

    def _on_card_selected(self, _event=None):
        label    = self._card_var.get()
        card_rec = self._card_map.get(label)
        if card_rec is None:
            return

        self._source_card_id = card_rec["id"]
        self._source_fid     = card_rec["fiber_id"]
        self._source_pid     = card_rec["polymer_id"]

        try:
            from core.services.service_transfer import resolve_constituent_props
            self._constituent_props = resolve_constituent_props(self._source_card_id)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self._win)
            return

        self._refresh_const_display()
        self._unlock_section2()

    def _refresh_const_display(self):
        for w in self._const_frame.winfo_children():
            w.destroy()

        props = self._constituent_props
        if not props:
            tk.Label(self._const_frame, text="No constituent properties found.",
                     font=FONT_STATUS).pack(anchor="w", pady=10)
            return

        _STAGES = [
            ("Stage 1 — Elastic",       ["matrix_modulus", "matrix_poisson"]),
            ("Stage 2 — Thermoelastic", ["f_cte1", "f_cte2", "m_cte"]),
            ("Stage 3 — Thermal",       ["k_f1", "k_f2", "k_m"]),
        ]

        r = 0
        for stage_lbl, keys in _STAGES:
            if r > 0:
                ttk.Separator(self._const_frame, orient="horizontal").grid(
                    row=r, column=0, columnspan=3, sticky="ew", pady=(6, 2))
                r += 1
            tk.Label(self._const_frame, text=stage_lbl,
                     font=("Helvetica", 10, "bold")).grid(
                row=r, column=0, columnspan=3, sticky="w", pady=(0, 3))
            r += 1

            for key in keys:
                meta     = _CONST_LABELS.get(key, key)
                unit     = _CONST_UNITS.get(key, "")
                row_data = props.get(key)

                if row_data is None:
                    val_txt  = "not inferred"
                    note_txt = "(datasheet will be used)"
                else:
                    val_txt  = f"{row_data['value']:.5g}"
                    if unit:
                        val_txt += f"  {unit}"
                    note_txt = row_data.get("note", "")

                tk.Label(self._const_frame, text=f"{meta}:",
                         font=FONT_SMALL, anchor="w", width=20).grid(
                    row=r, column=0, sticky="w", padx=(6, 4))
                tk.Label(self._const_frame, text=val_txt,
                         font=("Helvetica", 11, "bold"), anchor="w").grid(
                    row=r, column=1, sticky="w", padx=(0, 6))
                if note_txt:
                    tk.Label(self._const_frame, text=note_txt,
                             font=("Helvetica", 10), anchor="w").grid(
                        row=r, column=2, sticky="w")
                r += 1

    # ── Section ②: Microstructure (middle panel) ─────────────────────────────

    def _build_section2(self, panel: ttk.LabelFrame):
        panel.grid_columnconfigure(0, weight=1)

        # Mode radio
        mode_row = ttk.Frame(panel)
        mode_row.grid(row=0, column=0, sticky="w", pady=(0, 8))
        self._micro_mode = tk.StringVar(value="manual")

        rb_manual = ttk.Radiobutton(mode_row, text="Enter manually",
                                    variable=self._micro_mode, value="manual",
                                    command=self._on_mode_change, state="disabled")
        rb_manual.pack(side="left", padx=(0, 16))
        rb_infer  = ttk.Radiobutton(mode_row, text="Infer from measurements",
                                    variable=self._micro_mode, value="infer",
                                    command=self._on_mode_change, state="disabled")
        rb_infer.pack(side="left")
        self._mode_rbs = [rb_manual, rb_infer]

        # Container that swaps between manual and infer sub-sections
        self._mode_container = ttk.Frame(panel)
        self._mode_container.grid(row=1, column=0, sticky="ew")
        self._mode_container.grid_columnconfigure(0, weight=1)

        self._build_manual_pane()
        self._build_infer_pane()
        self._show_manual_pane()

        # Dummy locked_frame (not displayed — constituent props are in left panel)
        self._locked_frame = ttk.Frame(panel)

    def _build_manual_pane(self):
        self._manual_pane = ttk.Frame(self._mode_container)

        # 2-column layout: left 4 fields, right 3 fields
        # cols: 0=lbl_L 1=ent_L  3=lbl_R 4=ent_R
        fields = list(_MICRO_FIELDS)
        half   = (len(fields) + 1) // 2

        for i, field in enumerate(fields):
            lbl = _MICRO_LABELS[field]
            if i < half:
                lr, er = i, 0
            else:
                lr, er = i - half, 3

            tk.Label(self._manual_pane, text=f"{lbl}:", font=FONT_LABEL,
                     anchor="e", width=16).grid(row=lr, column=er, sticky="e",
                                                padx=(8 if er else 0, 6), pady=4)
            ent = tk.Entry(self._manual_pane, width=12, font=FONT_ENTRY,
                           state="disabled")
            ent.grid(row=lr, column=er + 1, sticky="w", pady=4)
            self._micro_entries[field] = ent

        # Gap column
        tk.Label(self._manual_pane, width=2).grid(row=0, column=2)

    def _build_infer_pane(self):
        self._infer_pane = ttk.Frame(self._mode_container)
        self._infer_pane.grid_columnconfigure(0, weight=1)

        # ── Section A: Microstructure Inputs ─────────────────────────────────
        tk.Label(self._infer_pane,
                 text="Microstructure inputs  (Fixed = use entered value,  Free = optimise)",
                 font=("Helvetica", 11, "bold"), fg="#333").grid(
            row=0, column=0, sticky="w", pady=(0, 2))

        micro_grid = ttk.Frame(self._infer_pane)
        micro_grid.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 4))

        # column headers
        for col, txt in [(0, "Field"), (1, "Mode"), (2, "Value / Initial guess"),
                         (3, "Bounds  (lo / hi)")]:
            tk.Label(micro_grid, text=txt, font=("Helvetica", 10, "bold"),
                     fg="gray").grid(row=0, column=col, sticky="w", padx=(4, 8), pady=(0, 2))

        self._micro_rows = {}
        for r, field in enumerate(_MICRO_FIELDS, start=1):
            row = MicroRow(micro_grid, r, field,
                           _MICRO_LABELS[field], _MICRO_BOUNDS[field])
            self._micro_rows[field] = row

        ttk.Separator(self._infer_pane, orient="horizontal").grid(
            row=2, column=0, sticky="ew", pady=(4, 6))

        # ── Section B: Target Composite Properties ────────────────────────────
        tk.Label(self._infer_pane,
                 text="Target composite properties  (check those you have measurements for):",
                 font=("Helvetica", 11, "bold"), fg="#333").grid(
            row=3, column=0, sticky="w", pady=(0, 2))

        tgt_grid = ttk.Frame(self._infer_pane)
        tgt_grid.grid(row=4, column=0, sticky="ew")

        # Two-column layout: left half | right half
        # Each half: label | checkbox | value | σ | unit
        for hcol, txt in [(0, "Field"), (1, ""), (2, "Value"), (3, "± σ"), (4, "Units"),
                          (6, "Field"), (7, ""), (8, "Value"), (9, "± σ"), (10, "Units")]:
            if txt:
                tk.Label(tgt_grid, text=txt,
                         font=("Helvetica", 10, "bold"), fg="gray").grid(
                    row=0, column=hcol, padx=4, pady=(0, 2))

        _DEFAULT_ACTIVE = {"E1", "E2", "nu12"}
        n    = len(_TARGET_FIELDS)
        half = (n + 1) // 2

        for i, (field, display) in enumerate(_TARGET_FIELDS):
            if i < half:
                base_col, data_row = 0, i + 1
            else:
                base_col, data_row = 6, (i - half) + 1

            tk.Label(tgt_grid, text=f"{display}:", font=FONT_LABEL,
                     anchor="e", width=6).grid(row=data_row, column=base_col,
                                               sticky="e", padx=(0, 2), pady=2)

            active_var = tk.BooleanVar(value=field in _DEFAULT_ACTIVE)
            ttk.Checkbutton(tgt_grid, variable=active_var,
                            command=lambda f=field: self._on_target_toggle(f)).grid(
                row=data_row, column=base_col + 1, padx=2)

            val_ent = tk.Entry(tgt_grid, width=10, font=FONT_ENTRY,
                               state="normal" if field in _DEFAULT_ACTIVE else "disabled")
            val_ent.grid(row=data_row, column=base_col + 2, padx=2, pady=2)

            sig_ent = tk.Entry(tgt_grid, width=7, font=FONT_ENTRY,
                               state="normal" if field in _DEFAULT_ACTIVE else "disabled")
            sig_ent.grid(row=data_row, column=base_col + 3, padx=2, pady=2)

            tk.Label(tgt_grid, text=UM.unit_label(field) or "MPa",
                     font=FONT_SMALL, fg="gray").grid(
                row=data_row, column=base_col + 4, padx=2, sticky="w")

            self._target_vars[field] = (active_var, val_ent, sig_ent)

        # gap column between target halves
        tk.Label(tgt_grid, text="  ", width=2).grid(row=1, column=5, rowspan=half)

        # ── Run Inverse button + status ───────────────────────────────────────
        btn_row = ttk.Frame(self._infer_pane)
        btn_row.grid(row=5, column=0, sticky="w", pady=(8, 0))
        self._infer_btn = ttk.Button(btn_row, text="Run Inverse",
                                     command=self._on_run_inverse,
                                     state="disabled")
        self._infer_btn.pack(side="left", padx=(0, 10))
        self._infer_status = tk.Label(btn_row, text="", font=FONT_STATUS, fg="gray")
        self._infer_status.pack(side="left")

    def _on_target_toggle(self, field: str):
        active, val_ent, sig_ent = self._target_vars[field]
        state = "normal" if active.get() else "disabled"
        val_ent.config(state=state)
        sig_ent.config(state=state)

    def _show_manual_pane(self):
        self._infer_pane.grid_remove()
        self._manual_pane.grid(row=0, column=0, sticky="ew")

    def _show_infer_pane(self):
        self._manual_pane.grid_remove()
        self._infer_pane.grid(row=0, column=0, sticky="ew")

    def _on_mode_change(self):
        if self._micro_mode.get() == "manual":
            self._show_manual_pane()
        else:
            self._show_infer_pane()

    def _unlock_section2(self):
        self._sec2.config(text="② Microstructure for New Printer")
        for rb in self._mode_rbs:
            rb.config(state="normal")
        for ent in self._micro_entries.values():
            ent.config(state="normal")
        self._infer_btn.config(state="normal")

        # Pre-fill manual + infer fields from source card's latest microstructure
        if self._source_card_id is not None:
            try:
                from core.services.service_cards import get_latest_microstructure
                latest = get_latest_microstructure(self._source_card_id)
                if latest:
                    mapping = {
                        "a11": "a11", "a22": "a22", "a12": "a12",
                        "a13": "a13", "a23": "a23", "ar": "ar",
                        "mf":  "fiber_massfrac",
                    }
                    for db_f, model_f in mapping.items():
                        val = latest.get(db_f)
                        if val is not None:
                            if model_f in self._micro_entries:
                                ent = self._micro_entries[model_f]
                                ent.delete(0, tk.END)
                                ent.insert(0, f"{float(val):.5g}")
                            if model_f in self._micro_rows:
                                self._micro_rows[model_f].set_value(float(val))
            except Exception:
                pass

        # Refresh locked constituent display
        self._refresh_locked_display()

        # Section ③ is always usable once a card is selected — the predict
        # button validates entries at click time, so unlock it now.
        self._unlock_section3(hint="Fill microstructure above, then predict.")

    def _refresh_locked_display(self):
        pass  # constituent props are shown in the left panel (_refresh_const_display)

    # ── Section ②: Run Inverse ────────────────────────────────────────────────

    def _on_run_inverse(self):
        if self._source_card_id is None:
            messagebox.showwarning("No card", "Select a source card first.",
                                   parent=self._win)
            return

        # Collect active targets
        targets: dict[str, float] = {}
        sigmas:  dict[str, float] = {}
        for field, (active, val_ent, sig_ent) in self._target_vars.items():
            if not active.get():
                continue
            raw = val_ent.get().strip()
            if not raw:
                messagebox.showerror("Missing target",
                                     f"Enter a value for {field} or deactivate it.",
                                     parent=self._win)
                return
            try:
                targets[field] = UM.from_display(field, float(raw))
            except ValueError:
                messagebox.showerror("Invalid value",
                                     f"{field} is not a valid number.",
                                     parent=self._win)
                return
            sig_raw = sig_ent.get().strip()
            if sig_raw:
                try:
                    sigmas[field] = UM.from_display(field, float(sig_raw))
                except ValueError:
                    pass

        if not targets:
            messagebox.showwarning("No targets",
                                   "Activate at least one target field.",
                                   parent=self._win)
            return

        # Snapshot micro-row state before handing off to the worker thread
        row_snapshot: dict = {}
        for field, micro_row in self._micro_rows.items():
            try:
                val = micro_row.get_value()
            except ValueError:
                messagebox.showerror("Invalid value",
                                     f"Microstructure field '{_MICRO_LABELS[field]}' "
                                     "has an invalid number.",
                                     parent=self._win)
                return
            try:
                bnd = micro_row.get_bounds()
            except ValueError:
                messagebox.showerror("Invalid bounds",
                                     f"Bounds for '{_MICRO_LABELS[field]}' "
                                     "are not valid numbers.",
                                     parent=self._win)
                return
            row_snapshot[field] = {
                "free":   micro_row.is_free(),
                "value":  val,
                "bounds": bnd,
            }

        self._infer_btn.config(state="disabled")
        self._infer_status.config(text="Running inverse…", fg="orange")

        threading.Thread(
            target=self._infer_worker,
            args=(targets, sigmas, row_snapshot),
            daemon=True,
        ).start()

    def _infer_worker(self, targets: dict, sigmas: dict,
                      row_snapshot: dict):
        """row_snapshot: {field: {'free': bool, 'value': float|None, 'bounds': (lo,hi)}}"""
        try:
            from core.services.service_transfer import prepare_transfer_inverse
            from core.services.service_inverse import run_inverse

            fixed_fields, free_fields, bounds, init_vals = prepare_transfer_inverse(
                self._source_card_id, row_snapshot
            )

            result = run_inverse(
                model_name="elastic",
                fixed_inputs=fixed_fields,
                free_inputs=free_fields,
                bounds=bounds if bounds else None,
                target_outputs=targets,
                sigmas=sigmas if sigmas else None,
                init_vals=init_vals if init_vals else None,
            )
            self._win.after(
                0,
                lambda r=result, ff=fixed_fields, fr=free_fields:
                    self._show_infer_results_dialog(r, ff, fr),
            )
        except Exception as exc:
            self._win.after(0, lambda exc=exc: self._on_infer_error(str(exc)))

    def _show_infer_results_dialog(self, result: dict,
                                    fixed_inputs: dict, free_fields: list):
        """Modal review dialog shown after inverse solve completes."""
        opt  = result.get("opt_free", {})
        pred = result.get("predicted_outputs", {})
        tgt  = result.get("target_outputs",    {})
        err  = result.get("final_error",        0.0)

        dlg = tk.Toplevel(self._win)
        dlg.title("Inverse Results — Review before continuing")
        dlg.geometry("720x620")
        dlg.minsize(640, 500)
        dlg.resizable(True, True)
        dlg.grab_set()
        dlg.lift()

        # ── scrollable content ────────────────────────────────────────────────
        canvas = tk.Canvas(dlg, borderwidth=0, highlightthickness=0)
        vsb    = ttk.Scrollbar(dlg, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="top", fill="both", expand=True)

        content = ttk.Frame(canvas, padding=(16, 12))
        cwin = canvas.create_window((0, 0), window=content, anchor="nw")
        content.bind("<Configure>",
                     lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(cwin, width=e.width))

        def _section(parent, title, row):
            tk.Label(parent, text=title,
                     font=("Helvetica", 13, "bold"), fg="#1a1a2e").grid(
                row=row, column=0, columnspan=4, sticky="w", pady=(10, 3))
            ttk.Separator(parent, orient="horizontal").grid(
                row=row + 1, column=0, columnspan=4, sticky="ew", pady=(0, 4))
            return row + 2

        def _row(parent, row, label, val_str, extra="", bold=False):
            font = ("Helvetica", 12, "bold") if bold else ("Helvetica", 12)
            tk.Label(parent, text=f"{label}:", font=font, anchor="e",
                     width=26).grid(row=row, column=0, sticky="e", padx=(0, 8))
            tk.Label(parent, text=val_str,   font=font, anchor="w").grid(
                row=row, column=1, sticky="w", padx=(0, 12))
            if extra:
                tk.Label(parent, text=extra, font=("Helvetica", 11), fg="gray",
                         anchor="w").grid(row=row, column=2, sticky="w")

        r = 0

        # ── 1. Constituent properties used (fixed) ────────────────────────────
        r = _section(content, "① Constituent Properties Used (fixed inputs)", r)
        _const_display = [
            ("matrix_modulus", "Matrix modulus",    "MPa"),
            ("matrix_poisson", "Matrix Poisson ν",  ""),
            ("f_cte1",         "Fiber CTE α₁₁",    "1/K"),
            ("f_cte2",         "Fiber CTE α₂₂",    "1/K"),
            ("m_cte",          "Matrix CTE",        "1/K"),
            ("matrix_CTE",     "Matrix CTE",        "1/K"),
            ("k_f1",           "Fiber k‖",          "W/m·K"),
            ("k_f2",           "Fiber k⊥",          "W/m·K"),
            ("k_m",            "Matrix k",          "W/m·K"),
        ]
        shown_const = set()
        for key, label, unit in _const_display:
            if key in fixed_inputs and label not in shown_const:
                shown_const.add(label)
                val = fixed_inputs[key]
                disp = f"{val:.5g}"
                src = ""
                cp = self._constituent_props.get(key) or self._constituent_props.get(
                    {"matrix_CTE": "m_cte", "f_CTE1": "f_cte1",
                     "f_CTE2": "f_cte2"}.get(key, ""))
                if isinstance(cp, dict):
                    src = f"[{cp.get('source_tag', '')}]"
                _row(content, r, label, f"{disp}  {unit}".strip(), src)
                r += 1
        if not shown_const:
            tk.Label(content, text="(no constituent properties found — using datasheet defaults)",
                     font=FONT_SMALL, fg="gray").grid(row=r, column=0, columnspan=4, sticky="w")
            r += 1

        # ── 2. Inferred microstructure ────────────────────────────────────────
        r = _section(content, "② Inferred Microstructure", r)
        _alias_canon = {"w_f": "fiber_massfrac", "ar_f": "ar"}
        shown_micro = set()
        for field, val in opt.items():
            canon = _alias_canon.get(field, field)
            label = _MICRO_LABELS.get(canon, canon)
            if label in shown_micro:
                continue
            shown_micro.add(label)
            _row(content, r, label, f"{float(val):.5g}", bold=True)
            r += 1
        if not shown_micro:
            tk.Label(content, text="(no free variables — nothing was optimised)",
                     font=FONT_SMALL, fg="gray").grid(row=r, column=0, columnspan=4, sticky="w")
            r += 1

        # ── 3. Predicted vs target outputs ────────────────────────────────────
        r = _section(content, "③ Predicted vs Target Outputs", r)
        # header
        for col, hdr in [(0, "Output"), (1, "Target"), (2, "Predicted"), (3, "Residual")]:
            tk.Label(content, text=hdr,
                     font=("Helvetica", 11, "bold"), fg="gray").grid(
                row=r, column=col, sticky="w", padx=(0, 12))
        r += 1

        for field, tval in tgt.items():
            pval   = pred.get(field, float("nan"))
            resid  = abs(float(pval) - float(tval)) if not (
                float(pval) != float(pval)) else float("nan")
            unit   = UM.unit_label(field)
            t_disp = UM.to_display(field, float(tval))
            p_disp = UM.to_display(field, float(pval))
            u_str  = f" {unit}" if unit else ""
            res_str = f"{resid:.4g}{u_str}" if resid == resid else "—"
            tk.Label(content, text=f"{field}:", font=FONT_LABEL, anchor="e",
                     width=8).grid(row=r, column=0, sticky="e", padx=(0, 8))
            tk.Label(content, text=f"{t_disp:.5g}{u_str}", font=FONT_LABEL,
                     anchor="w").grid(row=r, column=1, sticky="w", padx=(0, 12))
            tk.Label(content, text=f"{p_disp:.5g}{u_str}", font=FONT_LABEL,
                     fg="#007700", anchor="w").grid(row=r, column=2, sticky="w", padx=(0, 12))
            tk.Label(content, text=res_str, font=FONT_SMALL, fg="gray",
                     anchor="w").grid(row=r, column=3, sticky="w")
            r += 1

        # loss
        tk.Label(content, text=f"Final solver loss:  {err:.4e}",
                 font=("Helvetica", 12, "italic"), fg="#555").grid(
            row=r, column=0, columnspan=4, sticky="w", pady=(6, 0))
        r += 1

        # ── bottom bar: Continue / Re-run ─────────────────────────────────────
        ttk.Separator(dlg, orient="horizontal").pack(side="bottom", fill="x")
        bar = ttk.Frame(dlg, padding=(12, 8))
        bar.pack(side="bottom", fill="x")

        def _continue():
            dlg.destroy()
            self._accept_infer_result(result)

        ttk.Button(bar, text="Continue Transfer →",
                   command=_continue).pack(side="right", padx=(8, 0))
        ttk.Button(bar, text="Re-run / Adjust",
                   command=dlg.destroy).pack(side="right")

        self._infer_status.config(
            text=f"Solve done  (loss={err:.3e}) — review dialog",
            fg="#007700")
        self._infer_btn.config(state="normal")

    def _accept_infer_result(self, result: dict):
        """Called when user clicks 'Continue Transfer →' in the results dialog."""
        opt = result.get("opt_free", {})

        alias = {"w_f": "fiber_massfrac", "ar_f": "ar",
                 "fiber_massfrac": "fiber_massfrac", "ar": "ar"}
        for field, val in opt.items():
            canonical = alias.get(field, field)
            if canonical in self._micro_entries:
                ent = self._micro_entries[canonical]
                ent.delete(0, tk.END)
                ent.insert(0, f"{float(val):.5g}")
            if canonical in self._micro_rows:
                self._micro_rows[canonical].set_value(float(val))

        self._micro_inferred = True
        self._micro_mode.set("manual")
        self._show_manual_pane()
        self._unlock_section3(
            hint="Inferred microstructure applied. Select model and predict below.")

    def _on_infer_error(self, msg: str):
        self._infer_status.config(text="Inverse failed.", fg="red")
        self._infer_btn.config(state="normal")
        messagebox.showerror("Inverse Error", msg, parent=self._win)

    # ── Section ③: Predict & Save (right panel) ──────────────────────────────

    def _build_section3(self, panel: ttk.LabelFrame):
        panel.grid_columnconfigure(0, weight=1)

        # Model selector
        tk.Label(panel, text="Model:", font=FONT_LABEL).grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        model_row = ttk.Frame(panel)
        model_row.grid(row=1, column=0, sticky="w", pady=(0, 8))
        self._model_var = tk.StringVar(value="elastic")
        self._model_rbs = []
        for name in ("elastic", "thermoelastic", "thermal"):
            rb = ttk.Radiobutton(model_row, text=name.capitalize(),
                                 variable=self._model_var, value=name,
                                 state="disabled")
            rb.pack(side="left", padx=(0, 8))
            self._model_rbs.append(rb)

        # Predict button + status
        act_row = ttk.Frame(panel)
        act_row.grid(row=2, column=0, sticky="w", pady=(0, 8))
        self._predict_btn = ttk.Button(act_row, text="Predict",
                                       command=self._on_predict, state="disabled")
        self._predict_btn.pack(side="left", padx=(0, 10))
        self._pred_status = tk.Label(act_row, text="", font=FONT_STATUS, fg="gray")
        self._pred_status.pack(side="left")

        # Predicted outputs frame (expands vertically)
        self._outputs_frame = ttk.Frame(panel)
        self._outputs_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 8))
        panel.grid_rowconfigure(3, weight=1)
        tk.Label(self._outputs_frame,
                 text="(predictions will appear here)",
                 font=FONT_STATUS, fg="gray").pack(anchor="w")

        ttk.Separator(panel, orient="horizontal").grid(
            row=4, column=0, sticky="ew", pady=(4, 8))

        self._save_btn = ttk.Button(panel, text="Save to New Card…",
                                    command=self._on_save, state="disabled")
        self._save_btn.grid(row=5, column=0, sticky="w")

    def _unlock_section3(self, hint: str = ""):
        self._sec3.config(text="③ Predict & Save")
        for rb in self._model_rbs:
            rb.config(state="normal")
        self._predict_btn.config(state="normal")
        if hint:
            self._pred_status.config(text=hint, fg="gray")

    # ── predict ───────────────────────────────────────────────────────────────

    def _on_predict(self):
        micro = self._collect_microstructure()
        if micro is None:
            return
        model_name = self._model_var.get()
        self._predict_btn.config(state="disabled")
        self._pred_status.config(text="Running…", fg="orange")
        threading.Thread(
            target=self._predict_worker,
            args=(model_name, micro),
            daemon=True,
        ).start()

    def _predict_worker(self, model_name: str, micro: dict):
        try:
            from core.services.service_transfer import build_forward_inputs
            from core.services.service_forward import run_forward
            inputs  = build_forward_inputs(self._source_card_id, micro)
            outputs = run_forward(model_name, inputs)
            self._win.after(0, lambda: self._on_predict_done(model_name, inputs, outputs))
        except Exception as exc:
            self._win.after(0, lambda exc=exc: self._on_predict_error(str(exc)))

    def _on_predict_done(self, model_name: str, inputs: dict, outputs: dict):
        self._last_model_name  = model_name
        self._last_inputs      = inputs
        self._last_outputs     = outputs
        self._micro_inferred   = self._micro_inferred  # preserve flag

        # Refresh output display — format: "E1: 500.2 MPa" bold black
        for w in self._outputs_frame.winfo_children():
            w.destroy()

        for i, (field, val) in enumerate(outputs.items()):
            unit     = UM.unit_label(field)
            disp_val = UM.to_display(field, float(val))
            unit_str = f" {unit}" if unit else ""
            tk.Label(self._outputs_frame,
                     text=f"{field}:  {disp_val:.5g}{unit_str}",
                     font=FONT_BOLD, fg="black", anchor="w").pack(
                anchor="w", pady=1)

        self._pred_status.config(text=f"{model_name} done.", fg="#007700")
        self._predict_btn.config(state="normal")
        self._save_btn.config(state="normal")

    def _on_predict_error(self, msg: str):
        self._pred_status.config(text="Prediction failed.", fg="red")
        self._predict_btn.config(state="normal")
        messagebox.showerror("Prediction Error", msg, parent=self._win)

    # ── save ─────────────────────────────────────────────────────────────────

    def _on_save(self):
        if not self._last_outputs:
            messagebox.showwarning("No prediction",
                                   "Run a prediction first.",
                                   parent=self._win)
            return

        micro = self._collect_microstructure()
        if micro is None:
            return

        # Split inputs: microstructure → free_variables if inferred, else fixed
        micro_keys = {
            "a11", "a22", "a12", "a13", "a23",
            "ar", "ar_f", "fiber_massfrac", "w_f",
        }
        if self._micro_inferred:
            free_vars   = {k: v for k, v in self._last_inputs.items() if k in micro_keys}
            fixed_in    = {k: v for k, v in self._last_inputs.items() if k not in micro_keys}
        else:
            free_vars   = {}
            fixed_in    = dict(self._last_inputs)

        result = {
            "model":                 f"{self._last_model_name}_transfer",
            "free_variables":        free_vars,
            "fixed_inputs":          fixed_in,
            "predicted_outputs":     self._last_outputs,
            "target_outputs":        {},
            "sigmas":                {},
            "final_optimiser_error": 0.0,
            "solver":                {},
        }

        from gui_card_dialogs import SaveToCardDialog
        SaveToCardDialog(self._win, result,
                         fiber_id=self._source_fid,
                         polymer_id=self._source_pid)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _on_close(self):
        """Close the window and, in standalone mode, quit the hidden root too."""
        self._win.destroy()
        try:
            # If parent is a withdrawn Tk root (standalone mode), destroy it so
            # the Python process exits cleanly.
            if (isinstance(self._parent, tk.Tk)
                    and not self._parent.winfo_viewable()):
                self._parent.destroy()
        except Exception:
            pass

    def _collect_microstructure(self) -> Optional[dict[str, float]]:
        """Read manual entry fields; return dict or None on validation failure."""
        micro: dict[str, float] = {}
        for field, ent in self._micro_entries.items():
            raw = ent.get().strip()
            if not raw:
                messagebox.showerror(
                    "Missing field",
                    f"Microstructure field '{_MICRO_LABELS[field]}' is empty.\n"
                    "Fill all microstructure fields before predicting.",
                    parent=self._win)
                return None
            try:
                micro[field] = float(raw)
            except ValueError:
                messagebox.showerror("Invalid value",
                                     f"'{_MICRO_LABELS[field]}' is not a valid number.",
                                     parent=self._win)
                return None
        return micro


# ── helper: open from gui.py ─────────────────────────────────────────────────

def open_transfer_window(parent: tk.Widget) -> TransferWindow:
    return TransferWindow(parent)


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import subprocess
    import sys

    root = tk.Tk()
    root.withdraw()   # hide the empty root window
    TransferWindow(root)

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

    root.mainloop()
