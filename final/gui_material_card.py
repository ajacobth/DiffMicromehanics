"""gui_material_card.py — material card viewer.

Launchable from gui.py ("Material Card Viewer" button) or standalone:
    python gui_material_card.py

Tabs
----
  Summary              Card metadata + latest microstructure
  Constituent Props    Neat (web) vs. inferred — side-by-side table
  Microstructure       Full snapshot history with per-field provenance
  Composite Props      Predicted and experimental properties
  Inference History    All runs; click a row to expand full I/O
"""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import db.db as _db

FONT_TITLE = ("Helvetica", 15, "bold")
FONT_BOLD  = ("Helvetica", 13, "bold")
FONT_LABEL = ("Helvetica", 12)
FONT_SMALL = ("Helvetica", 10)
FONT_MONO  = ("Courier",   11)
FONT_STATUS = ("Helvetica", 12, "italic")


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.5g}"
    except Exception:
        return str(v)


def _scrolled_tree(parent, columns: tuple, col_widths: dict | None = None,
                   height: int = 14) -> tuple[ttk.Treeview, ttk.Scrollbar]:
    """Return (tree, vsb) packed into parent with a vertical scrollbar."""
    parent.grid_columnconfigure(0, weight=1)
    tree = ttk.Treeview(parent, columns=columns, show="headings", height=height)
    for c in columns:
        tree.heading(c, text=c)
        w = (col_widths or {}).get(c, 110)
        tree.column(c, width=w, stretch=True)
    vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    return tree, vsb


# ══════════════════════════════════════════════════════════════════════════════
# MaterialCardViewer
# ══════════════════════════════════════════════════════════════════════════════

class MaterialCardViewer:
    """Toplevel material card viewer window."""

    def __init__(self, parent: tk.Widget):
        self._win = tk.Toplevel(parent)
        self._win.title("Material Card Viewer")
        self._win.geometry("960x720")
        self._win.minsize(720, 500)
        self._win.grid_rowconfigure(1, weight=1)
        self._win.grid_columnconfigure(0, weight=1)

        self._card_map: dict[str, int] = {}  # display label → print_config id
        self._run_data: dict[str, dict] = {}  # tree iid → run dict

        self._build()
        self._load_cards()

    # ── top bar ───────────────────────────────────────────────────────────────

    def _build(self):
        hdr = ttk.Frame(self._win, padding=(12, 8))
        hdr.grid(row=0, column=0, sticky="ew")

        tk.Label(hdr, text="Material Card Viewer",
                 font=FONT_TITLE).grid(row=0, column=0, sticky="w", padx=(0, 20))

        tk.Label(hdr, text="Card:", font=FONT_LABEL).grid(
            row=0, column=1, padx=(0, 4))
        self._card_var = tk.StringVar()
        self._card_cb  = ttk.Combobox(hdr, textvariable=self._card_var,
                                       state="readonly", width=46, font=FONT_LABEL)
        self._card_cb.grid(row=0, column=2, padx=4)
        self._card_cb.bind("<<ComboboxSelected>>", lambda _: self._load_card())

        ttk.Button(hdr, text="Refresh",
                   command=self._load_card).grid(row=0, column=3, padx=(8, 0))

        ttk.Button(hdr, text="Export JSON",
                   command=self._export_json).grid(row=0, column=4, padx=(8, 0))

        ttk.Separator(self._win, orient="horizontal").grid(
            row=0, column=0, sticky="ew", pady=(48, 0))

        # Notebook
        self._nb = ttk.Notebook(self._win)
        self._nb.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        self._tab_summary     = ttk.Frame(self._nb, padding=10)
        self._tab_constituent = ttk.Frame(self._nb, padding=10)
        self._tab_micro       = ttk.Frame(self._nb, padding=10)
        self._tab_composite   = ttk.Frame(self._nb, padding=10)
        self._tab_history     = ttk.Frame(self._nb, padding=10)

        self._nb.add(self._tab_summary,     text="Summary")
        self._nb.add(self._tab_constituent, text="Constituent Properties")
        self._nb.add(self._tab_micro,       text="Microstructure")
        self._nb.add(self._tab_composite,   text="Composite Properties")
        self._nb.add(self._tab_history,     text="Inference History")

        tk.Label(self._tab_summary,
                 text="Select a card above to view.",
                 font=FONT_STATUS, fg="gray").pack(padx=20, pady=30)

    # ── card index ────────────────────────────────────────────────────────────

    def _load_cards(self):
        try:
            cards = _db.get_all_print_configs()
        except Exception as exc:
            messagebox.showerror("DB Error", str(exc), parent=self._win)
            return
        self._card_map = {}
        for c in cards:
            f = _db.get_fiber(c["fiber_id"])
            p = _db.get_polymer(c["polymer_id"])
            f_name = f["name"] if f else f"fiber:{c['fiber_id']}"
            p_name = p["name"] if p else f"polymer:{c['polymer_id']}"
            display = f"{c['name']}  [{f_name} / {p_name}]"
            self._card_map[display] = c["id"]
        self._card_cb["values"] = list(self._card_map)

    # ── load & render ─────────────────────────────────────────────────────────

    def _load_card(self):
        cfg_id = self._card_map.get(self._card_var.get())
        if cfg_id is None:
            return
        try:
            card = _db.get_print_config_card(cfg_id)
        except Exception as exc:
            messagebox.showerror("DB Error", str(exc), parent=self._win)
            return
        self._render_summary(card)
        self._render_constituent(card)
        self._render_micro(card)
        self._render_composite(card)
        self._render_history(card)

    @staticmethod
    def _clear(tab: ttk.Frame):
        for w in tab.winfo_children():
            w.destroy()
        for i in range(10):
            tab.grid_rowconfigure(i, weight=0)

    # ── tab: Summary ──────────────────────────────────────────────────────────

    def _render_summary(self, card: dict):
        self._clear(self._tab_summary)
        tab = self._tab_summary
        tab.grid_columnconfigure(1, weight=1)

        cfg     = card["config"]   or {}
        fiber   = card["fiber"]    or {}
        polymer = card["polymer"]  or {}
        printer = card["printer"]  or {}
        micro   = card["microstructure"]

        rows = [
            ("Card name",   cfg.get("name", "—")),
            ("Created",     (cfg.get("created_at") or "")[:19]),
            ("Fiber",       fiber.get("name", "—")),
            ("Polymer",     polymer.get("name", "—")),
            ("Printer",     printer.get("name", "—")),
            ("Notes",       cfg.get("notes", "")),
        ]
        if micro:
            rows += [
                ("── Microstructure ──", ""),
                ("Mass fraction (mf)", _fmt(micro.get("mf"))),
                ("Aspect ratio (ar)",  _fmt(micro.get("ar"))),
                ("a11",                _fmt(micro.get("a11"))),
                ("a22",                _fmt(micro.get("a22"))),
                ("a12",                _fmt(micro.get("a12"))),
                ("a13",                _fmt(micro.get("a13"))),
                ("a23",                _fmt(micro.get("a23"))),
            ]

        pc = card.get("processing_condition")
        rows += [("── Printing conditions ──", "")]
        if pc:
            if pc.get("bead_width")      is not None: rows.append(("Bead width",      f"{pc['bead_width']} mm"))
            if pc.get("bead_height")     is not None: rows.append(("Bead height",      f"{pc['bead_height']} mm"))
            if pc.get("nozzle_diameter") is not None: rows.append(("Nozzle diameter",  f"{pc['nozzle_diameter']} mm"))
            if pc.get("speed")           is not None: rows.append(("Print speed",      f"{pc['speed']} mm/s"))
            if pc.get("notes"):                       rows.append(("Notes",            pc["notes"]))
        else:
            rows.append(("", "(none recorded)"))

        for i, (k, v) in enumerate(rows):
            is_hdr = k.startswith("──")
            fg = "gray" if is_hdr else "black"
            fw = FONT_BOLD if is_hdr else FONT_LABEL
            tk.Label(tab, text=k + ("" if is_hdr else ":"),
                     font=fw, fg=fg, anchor="e").grid(
                row=i, column=0, sticky="e", padx=(0, 12), pady=3)
            tk.Label(tab, text=str(v), font=FONT_LABEL, anchor="w").grid(
                row=i, column=1, sticky="w", pady=3)

        base = len(rows)
        for j, (lbl, val) in enumerate([
            ("Inference runs",            len(card.get("inference_runs", []))),
            ("Experimental measurements", len(card.get("experimental_measurements", []))),
            ("Composite properties",      len(card.get("composite_properties", []))),
        ]):
            tk.Label(tab, text=lbl + ":", font=FONT_LABEL, fg="gray",
                     anchor="e").grid(row=base + j, column=0,
                                      sticky="e", padx=(0, 12), pady=(10 if j == 0 else 3))
            tk.Label(tab, text=str(val), font=FONT_BOLD).grid(
                row=base + j, column=1, sticky="w",
                pady=(10 if j == 0 else 3))

    # ── tab: Constituent Properties ───────────────────────────────────────────

    def _render_constituent(self, card: dict):
        self._clear(self._tab_constituent)
        tab = self._tab_constituent
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        tk.Label(tab, text="Neat (web) vs. inferred — most recent inferred per property",
                 font=FONT_SMALL, fg="gray").grid(row=0, column=0,
                                                   columnspan=2, sticky="w", pady=(0, 4))

        cols = ("Component", "Property", "Neat / Web", "Inferred",
                "Source", "Run ID", "Date")
        widths = {"Component": 80, "Property": 130, "Neat / Web": 110,
                  "Inferred": 110, "Source": 90, "Run ID": 60, "Date": 100}
        tree, vsb = _scrolled_tree(tab, cols, widths, height=16)
        tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

        fiber   = card["fiber"]   or {}
        polymer = card["polymer"] or {}

        neat_fiber: dict[str, object] = {
            "e1":            fiber.get("neat_E1"),
            "e2":            fiber.get("neat_E2"),
            "g12":           fiber.get("neat_G12"),
            "f_nu12":        fiber.get("neat_nu12"),
            "f_nu23":        fiber.get("neat_nu23"),
            "fiber_density": fiber.get("neat_rho"),
            "k_f1":          fiber.get("neat_k1"),
            "k_f2":          fiber.get("neat_k2"),
            "f_CTE1":        fiber.get("neat_CTE1"),
            "f_CTE2":        fiber.get("neat_CTE2"),
        }
        neat_poly: dict[str, object] = {
            "matrix_modulus": polymer.get("neat_E1"),
            "matrix_poisson": polymer.get("neat_nu12"),
            "matrix_density": polymer.get("neat_rho"),
            "k_m":            polymer.get("neat_k"),
            "matrix_CTE":     polymer.get("neat_CTE"),
        }

        # Most recent inferred value per property name
        def _best(prop_list) -> dict[str, dict]:
            seen: dict[str, dict] = {}
            for p in prop_list:  # list is newest-first
                if p["property_name"] not in seen:
                    seen[p["property_name"]] = p
            return seen

        inf_f = _best(card["constituent_properties"]["fiber"])
        inf_p = _best(card["constituent_properties"]["polymer"])

        all_f = sorted(set(list(neat_fiber) + list(inf_f)))
        all_p = sorted(set(list(neat_poly)  + list(inf_p)))

        for prop in all_f:
            inf = inf_f.get(prop)
            tree.insert("", "end", values=(
                "Fiber", prop,
                _fmt(neat_fiber.get(prop)),
                _fmt(inf["value"] if inf else None),
                inf["source_tag"] if inf else "—",
                inf["inference_run_id"] if inf else "—",
                (inf.get("created_at") or "")[:10] if inf else "—",
            ))
        for prop in all_p:
            inf = inf_p.get(prop)
            tree.insert("", "end", values=(
                "Polymer", prop,
                _fmt(neat_poly.get(prop)),
                _fmt(inf["value"] if inf else None),
                inf["source_tag"] if inf else "—",
                inf["inference_run_id"] if inf else "—",
                (inf.get("created_at") or "")[:10] if inf else "—",
            ))

    # ── tab: Microstructure ───────────────────────────────────────────────────

    def _render_micro(self, card: dict):
        self._clear(self._tab_micro)
        tab = self._tab_micro
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        tk.Label(tab, text="All snapshots — newest first.  "
                            "Provenance column shows source per field.",
                 font=FONT_SMALL, fg="gray").grid(row=0, column=0,
                                                   columnspan=2, sticky="w", pady=(0, 4))

        cfg_id = card["config"]["id"]
        try:
            snaps = _db.get_all_microstructure_snapshots(cfg_id)
        except Exception:
            snaps = []

        cols = ("Date", "mf", "ar", "a11", "a22", "a12", "a13", "a23",
                "Provenance")
        widths = {c: 72 for c in cols}
        widths["Date"] = 130
        widths["Provenance"] = 220
        tree, vsb = _scrolled_tree(tab, cols, widths, height=14)
        tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

        for snap in snaps:
            prov = snap.get("provenance") or {}
            prov_str = ", ".join(
                f"{k}:{v}" for k, v in prov.items() if v
            ) or "—"
            tree.insert("", "end", values=(
                (snap.get("created_at") or "")[:16],
                _fmt(snap.get("mf")),
                _fmt(snap.get("ar")),
                _fmt(snap.get("a11")),
                _fmt(snap.get("a22")),
                _fmt(snap.get("a12")),
                _fmt(snap.get("a13")),
                _fmt(snap.get("a23")),
                prov_str,
            ))

    # ── tab: Composite Properties ─────────────────────────────────────────────

    def _render_composite(self, card: dict):
        self._clear(self._tab_composite)
        tab = self._tab_composite
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        tk.Label(tab,
                 text="Predicted (from surrogate) and experimental measurements.  "
                      "Sorted newest first.",
                 font=FONT_SMALL, fg="gray").grid(row=0, column=0, columnspan=2,
                                                   sticky="w", pady=(0, 4))

        cols = ("Property", "Value", "Unit", "Source", "T (°C)", "Date",
                "Run ID")
        widths = {"Property": 120, "Value": 120, "Unit": 80, "Source": 110,
                  "T (°C)": 70, "Date": 105, "Run ID": 60}
        tree, vsb = _scrolled_tree(tab, cols, widths, height=14)
        tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

        for row in card.get("composite_properties", []):
            tree.insert("", "end", values=(
                row["property_name"],
                _fmt(row["value"]),
                row.get("unit") or "—",
                row.get("source_tag") or "—",
                _fmt(row.get("temperature_C")),
                (row.get("created_at") or "")[:10],
                row.get("inference_run_id") or "—",
            ))

        for row in card.get("experimental_measurements", []):
            unc = row.get("uncertainty")
            val_str = _fmt(row["value"])
            if unc is not None:
                val_str += f" ± {_fmt(unc)}"
            tree.insert("", "end", values=(
                row["property_name"],
                val_str,
                row.get("unit") or "—",
                "experimental",
                _fmt(row.get("temperature_C")),
                row.get("date") or "—",
                "—",
            ))

        def _on_composite_right_click(event):
            iid = tree.identify_row(event.y)
            if not iid:
                return
            vals = tree.item(iid, "values")
            if not vals:
                return
            prop_name = vals[0]
            source    = vals[3]
            cfg_id    = self._card_map.get(self._card_var.get())
            if cfg_id is None:
                return
            menu = tk.Menu(self._win, tearoff=0)
            menu.add_command(
                label=f"Set '{prop_name}' preferred source → '{source}'",
                command=lambda: self._set_pref(cfg_id, prop_name, source),
            )
            menu.add_command(
                label=f"Clear preference for '{prop_name}'",
                command=lambda: self._clear_pref(cfg_id, prop_name),
            )
            menu.tk_popup(event.x_root, event.y_root)

        tree.bind("<Button-2>", _on_composite_right_click)   # macOS right-click
        tree.bind("<Button-3>", _on_composite_right_click)   # Windows/Linux

    # ── tab: Inference History ────────────────────────────────────────────────

    def _render_history(self, card: dict):
        self._clear(self._tab_history)
        tab = self._tab_history
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        tk.Label(tab, text="Click a row to expand inputs / outputs.",
                 font=FONT_SMALL, fg="gray").grid(row=0, column=0,
                                                   columnspan=2, sticky="w", pady=(0, 4))

        cols = ("ID", "Stage", "Loss", "Date", "Snap ID", "Notes")
        widths = {"ID": 50, "Stage": 140, "Loss": 90, "Date": 140,
                  "Snap ID": 65, "Notes": 200}
        tree, vsb = _scrolled_tree(tab, cols, widths, height=8)
        tree.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")

        detail_lf = ttk.LabelFrame(tab, text="Run details", padding=6)
        detail_lf.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        detail_lf.grid_columnconfigure(0, weight=1)

        detail = tk.Text(detail_lf, height=9, font=FONT_MONO, state="disabled",
                         wrap="none", bg="#1e1e2e", fg="#cdd6f4")
        hsb = ttk.Scrollbar(detail_lf, orient="horizontal",
                             command=detail.xview)
        detail.configure(xscrollcommand=hsb.set)
        detail.grid(row=0, column=0, sticky="ew")
        hsb.grid(row=1, column=0, sticky="ew")

        self._run_data = {}
        for run in card.get("inference_runs", []):
            iid = tree.insert("", "end", values=(
                run["id"],
                run.get("stage") or "—",
                f"{run['loss']:.3e}" if run.get("loss") is not None else "—",
                (run.get("created_at") or "")[:16],
                run.get("microstructure_snap_id") or "—",
                run.get("notes") or "",
            ))
            self._run_data[iid] = run

        def _on_select(event):
            sel = tree.selection()
            if not sel:
                return
            run = self._run_data.get(sel[0])
            if not run:
                return
            lines = [
                f"Run ID: {run['id']}   Stage: {run.get('stage')}   "
                f"Loss: {run.get('loss')}",
            ]
            inp = run.get("inputs") or run.get("inputs_json")
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    pass
            out = run.get("outputs") or run.get("outputs_json")
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except Exception:
                    pass

            if isinstance(inp, dict):
                lines.append("\nInputs:")
                for k, v in inp.items():
                    lines.append(f"  {k:<30}: {_fmt(v)}")
            if isinstance(out, dict):
                lines.append("\nOutputs:")
                for k, v in out.items():
                    lines.append(f"  {k:<30}: {_fmt(v)}")

            detail.config(state="normal")
            detail.delete("1.0", tk.END)
            detail.insert(tk.END, "\n".join(lines))
            detail.config(state="disabled")

        tree.bind("<<TreeviewSelect>>", _on_select)

    # ── export / preferences helpers ──────────────────────────────────────────

    def _export_json(self):
        cfg_id = self._card_map.get(self._card_var.get())
        if cfg_id is None:
            messagebox.showwarning("No card", "Select a card first.", parent=self._win)
            return
        from tkinter import filedialog
        import json as _json
        path = filedialog.asksaveasfilename(
            parent=self._win,
            title="Export card as JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"card_{cfg_id}.json",
        )
        if not path:
            return
        try:
            card = _db.get_print_config_card(cfg_id)
            with open(path, "w") as f:
                _json.dump(card, f, indent=2, default=str)
            messagebox.showinfo("Exported", f"Card exported to:\n{path}", parent=self._win)
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc), parent=self._win)

    def _set_pref(self, cfg_id: int, prop_name: str, source: str):
        try:
            _db.set_property_preference(cfg_id, prop_name, source)
            messagebox.showinfo("Preference set",
                                f"'{prop_name}' will now prefer '{source}'.",
                                parent=self._win)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self._win)

    def _clear_pref(self, cfg_id: int, prop_name: str):
        try:
            with _db._connect() as conn:
                conn.execute(
                    "DELETE FROM property_preferences WHERE print_config_id=? AND property_name=?",
                    (cfg_id, prop_name),
                )
            messagebox.showinfo("Preference cleared",
                                f"Preference for '{prop_name}' removed.",
                                parent=self._win)
        except Exception as exc:
            messagebox.showerror("Error", str(exc), parent=self._win)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.withdraw()   # hide the empty root window
    viewer = MaterialCardViewer(root)
    viewer._win.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
