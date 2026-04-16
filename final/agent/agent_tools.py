"""
All tools available to the DiffMicromechanics agent.

Each tool is a @tool-decorated function. Add new tools here as capabilities
are built out (elastic inverse, save to card, etc.).

Current tools:
    search_knowledge_base    — RAG over agent/knowledge/ PDFs
    list_materials           — list all fibers, polymers, printers in the DB
    get_material_details     — full datasheet for one fiber or polymer by name
    list_cards               — list all material cards and their stage status
    get_card_status          — full detail view of one material card
    get_model_inputs_outputs — field names + units for elastic/thermoelastic models
    predict_properties       — forward prediction (elastic + thermoelastic)
    add_fiber                — add a new fiber to the material library
    add_polymer              — add a new polymer to the material library
    check_identifiability    — FIM analysis: can these measurements identify these unknowns?
"""

import re
import sys
from pathlib import Path

# Allow imports from final/ when running as agent
_FINAL = Path(__file__).parent.parent
if str(_FINAL) not in sys.path:
    sys.path.insert(0, str(_FINAL))

from langchain_core.tools import tool

from agent.rag import search_knowledge_base
import core.services.service_material as _smat
import core.services.service_cards as _scards
import core.services.service_forward as _sfwd
import core.services.service_fim as _sfim


# ── Module-level constants for forward/FIM tools ──────────────────────────────

_ELASTIC_REQUIRED = frozenset([
    "e1", "e2", "g12", "f_nu12", "f_nu23", "ar", "fiber_massfrac",
    "fiber_density", "matrix_modulus", "matrix_poisson", "matrix_density",
    "a11", "a22", "a12", "a13", "a23",
])

_TE_EXTRA_REQUIRED = frozenset(["f_cte1", "f_cte2", "m_cte"])

# Synonym maps for check_identifiability — maps user terms → canonical model field names
_FREE_SYNONYMS = {
    "a11": "a11", "a22": "a22", "a12": "a12", "a13": "a13", "a23": "a23",
    "mf": "fiber_massfrac", "w_f": "fiber_massfrac", "wf": "fiber_massfrac",
    "massfrac": "fiber_massfrac", "fiber_mass_fraction": "fiber_massfrac",
    "mass_fraction": "fiber_massfrac", "fiber_massfrac": "fiber_massfrac",
    "ar": "ar", "ar_f": "ar", "aspect_ratio": "ar",
    "matrix_modulus": "matrix_modulus", "em": "matrix_modulus",
    "matrix_e": "matrix_modulus", "matrix_stiffness": "matrix_modulus",
    "matrix_poisson": "matrix_poisson", "nu_m": "matrix_poisson",
    "f_cte1": "f_cte1", "fiber_cte1": "f_cte1", "alpha_f1": "f_cte1",
    "f_cte2": "f_cte2", "fiber_cte2": "f_cte2", "alpha_f2": "f_cte2",
    "m_cte": "m_cte", "matrix_cte": "m_cte", "alpha_m": "m_cte",
}

# Maps user measurement names → canonical model output field names
_MEAS_SYNONYMS = {
    "E1": "E1", "e1": "E1", "E2": "E2", "e2": "E2", "E3": "E3", "e3": "E3",
    "G12": "G12", "g12": "G12", "G13": "G13", "g13": "G13",
    "G23": "G23", "g23": "G23",
    "nu12": "nu12", "nu13": "nu13", "nu23": "nu23", "poisson": "nu12",
    "CTE11": "CTE11", "cte11": "CTE11", "alpha11": "CTE11",
    "CTE22": "CTE22", "cte22": "CTE22", "alpha22": "CTE22",
    "CTE33": "CTE33", "cte33": "CTE33",
    "CTE12": "CTE12", "CTE13": "CTE13", "CTE23": "CTE23",
}

_DEFAULT_BOUNDS = {
    "a11":            (0.20,    0.85),
    "a22":            (0.01,    0.40),
    "a12":            (-0.10,   0.10),
    "a13":            (-0.10,   0.10),
    "a23":            (-0.10,   0.10),
    "fiber_massfrac": (0.05,    0.60),
    "ar":             (5.0,    100.0),
    "matrix_modulus": (1500.0, 6000.0),
    "matrix_poisson": (0.28,    0.45),
    "f_cte1":         (-2e-6,   5e-6),
    "f_cte2":         (5e-6,   30e-6),
    "m_cte":          (30e-6, 120e-6),
}

# Fallback nominal inputs when no fiber/polymer/card is provided
_NOMINAL_INPUTS = {
    "e1": 240000.0, "e2": 15000.0, "g12": 28000.0,
    "f_nu12": 0.2, "f_nu23": 0.4,
    "ar": 20.0, "fiber_massfrac": 0.20,
    "fiber_density": 1780.0, "rho_f": 1780.0,
    "matrix_modulus": 3100.0, "matrix_poisson": 0.37,
    "matrix_density": 1280.0, "rho_m": 1280.0,
    "a11": 0.60, "a22": 0.15, "a12": 0.0, "a13": 0.0, "a23": 0.0,
    "f_cte1": -0.5e-6, "f_cte2": 15.0e-6, "m_cte": 60.0e-6,
}


# ── Discovery tools ────────────────────────────────────────────────────────────

@tool
def list_materials() -> str:
    """
    List the raw material library: all fibers, polymers, and printers registered
    in the database, with their datasheet properties and IDs.
    Use this to find fiber_id, polymer_id, or printer_id before running a solver.
    Do NOT call this when the user asks about material cards or characterization
    status — use list_cards for that.
    """
    try:
        fibers   = _smat.list_fibers()
        polymers = _smat.list_polymers()
        printers = _smat.list_printers()
    except Exception as e:
        return f"Database error: {e}"

    lines = ["AVAILABLE MATERIALS\n"]

    lines.append("Fibers:")
    if fibers:
        for f in fibers:
            lines.append(f"  id={f['id']}  {f['name']}  supplier={f.get('supplier', '?')}")
            _fiber_props = [
                ("E1",      f.get("neat_E1"),   "MPa"),
                ("E2",      f.get("neat_E2"),   "MPa"),
                ("G12",     f.get("neat_G12"),  "MPa"),
                ("nu12",    f.get("neat_nu12"), ""),
                ("nu23",    f.get("neat_nu23"), ""),
                ("density", f.get("neat_rho"),  "kg/m³"),
                ("k1",      f.get("neat_k1"),   "W/m·K"),
                ("k2",      f.get("neat_k2"),   "W/m·K"),
            ]
            for label, val, unit in _fiber_props:
                if val is not None:
                    lines.append(f"    {label} = {val} {unit}".rstrip())
    else:
        lines.append("  (none)")

    lines.append("\nPolymers:")
    if polymers:
        for p in polymers:
            lines.append(f"  id={p['id']}  {p['name']}  supplier={p.get('supplier', '?')}")
            _poly_props = [
                ("E (matrix modulus)", p.get("neat_E1"),   "MPa"),
                ("nu12",               p.get("neat_nu12"), ""),
                ("density",            p.get("neat_rho"),  "kg/m³"),
                ("k (conductivity)",   p.get("neat_k"),    "W/m·K"),
            ]
            for label, val, unit in _poly_props:
                if val is not None:
                    lines.append(f"    {label} = {val} {unit}".rstrip())
    else:
        lines.append("  (none)")

    lines.append("\nPrinters:")
    if printers:
        for pr in printers:
            lines.append(f"  id={pr['id']}  {pr['name']} ({pr.get('manufacturer','?')})")
    else:
        lines.append("  (none — add via db.add_printer() or ask the user to register one)")

    return "\n".join(lines)


@tool
def list_cards() -> str:
    """
    List all material cards saved in the database. A material card is a
    (fiber + polymer + printer) combination that has been characterized.
    Shows which of the four characterization stages (elastic, thermoelastic,
    thermal, transfer) are complete for each card.
    Call this when the user asks: "what cards do we have", "what has been
    characterized", "show me the cards", or "what work has been done".
    """
    try:
        configs  = _smat.list_cards()
        fibers   = {f["id"]: f for f in _smat.list_fibers()}
        polymers = {p["id"]: p for p in _smat.list_polymers()}
        printers = {pr["id"]: pr for pr in _smat.list_printers()}
    except Exception as e:
        return f"Database error: {e}"

    if not configs:
        return "No material cards exist yet. Run Stage 1 (elastic inverse) to create the first card."

    lines = ["MATERIAL CARDS\n"]
    for cfg in configs:
        fiber   = fibers.get(cfg["fiber_id"],   {})
        polymer = polymers.get(cfg["polymer_id"], {})

        printer_name = "no printer"
        if cfg.get("printer_id"):
            pr = printers.get(cfg["printer_id"], {})
            printer_name = pr.get("name", f"printer_id={cfg['printer_id']}")

        stages = _smat.get_completed_stages(cfg["id"], cfg["fiber_id"], cfg["polymer_id"])
        stage_str = ", ".join(stages) if stages else "none"

        fiber_name   = fiber.get("name",   f"fiber_id={cfg['fiber_id']}")
        polymer_name = polymer.get("name", f"polymer_id={cfg['polymer_id']}")

        lines.append(
            f"  card_id={cfg['id']}  \"{cfg['name']}\"\n"
            f"    Fiber:   {fiber_name}\n"
            f"    Polymer: {polymer_name}\n"
            f"    Printer: {printer_name}\n"
            f"    Completed stages: {stage_str}\n"
            f"    Created: {cfg.get('created_at','?')}"
        )

    return "\n".join(lines)


@tool
def get_card_status(card_id: int) -> str:
    """
    Get the full characterization status of a material card: which stages are
    complete, all constituent properties stored (with provenance — whether each
    value was inputted from a datasheet or inferred by the solver), and the
    current microstructure snapshot.

    Use this before recommending the next stage to run, or when the user asks
    what is known about a specific material card.
    """
    try:
        card = _scards.load_card(card_id)
    except ValueError as e:
        return f"Card not found: {e}"
    except Exception as e:
        return f"Database error: {e}"

    cfg     = card["config"]
    fiber   = card["fiber"]   or {}
    polymer = card["polymer"] or {}
    printer = card["printer"] or {}
    micro   = card["microstructure"]

    fiber_name   = fiber.get("name",   f"fiber_id={cfg['fiber_id']}")
    polymer_name = polymer.get("name", f"polymer_id={cfg['polymer_id']}")
    printer_name = printer.get("name", "no printer")

    stages = _smat.get_completed_stages(card_id, cfg["fiber_id"], cfg["polymer_id"])

    lines = [
        f"CARD STATUS — \"{cfg['name']}\"  (card_id={card_id})",
        f"  Fiber:   {fiber_name}",
        f"  Polymer: {polymer_name}",
        f"  Printer: {printer_name}",
        f"  Completed stages: {', '.join(stages) if stages else 'none — no inverse runs saved yet'}",
    ]

    # ── Microstructure ─────────────────────────────────────────────────────────
    lines.append("\nMicrostructure (latest snapshot):")
    if micro:
        prov = micro.get("provenance") or {}
        fields = [("a11", "fiber alignment x"), ("a22", "transverse y"),
                  ("a12", "shear xy"),          ("a13", "shear xz"),
                  ("a23", "shear yz"),           ("mf",  "fiber mass fraction"),
                  ("ar",  "aspect ratio")]
        for key, label in fields:
            val = micro.get(key)
            if val is not None:
                src = prov.get(key, "unknown")
                lines.append(f"  {key:4s} = {val:.4f}  ({label})  [{src}]")
    else:
        lines.append("  (no microstructure stored — Stage 1 not yet run or not saved)")

    # ── Constituent properties ─────────────────────────────────────────────────
    lines.append("\nConstituent properties:")

    fiber_props   = card["constituent_properties"].get("fiber",   [])
    polymer_props = card["constituent_properties"].get("polymer", [])

    # Deduplicate: keep newest per property_name
    def _dedup(props: list) -> dict:
        seen: dict = {}
        for p in props:  # already newest-first from db
            n = p["property_name"]
            if n not in seen:
                seen[n] = p
        return seen

    fp = _dedup(fiber_props)
    pp = _dedup(polymer_props)

    _UNITS = {
        "matrix_modulus": "MPa",  "matrix_poisson": "",
        "f_cte1": "1/K",          "f_cte2": "1/K",  "m_cte": "1/K",
        "k_f1": "W/m·K",          "k_f2": "W/m·K",  "k_m": "W/m·K",
        "p1": "W/m·K",            "p2": "W/m·K",
    }

    if fp:
        lines.append(f"  Fiber ({fiber_name}):")
        for name, p in sorted(fp.items()):
            unit = _UNITS.get(name, "")
            unit_str = f" {unit}" if unit else ""
            lines.append(f"    {name} = {p['value']:.6g}{unit_str}  [{p['source_tag']}]")
    else:
        lines.append(f"  Fiber ({fiber_name}): (datasheet values only — nothing inferred)")

    if pp:
        lines.append(f"  Polymer ({polymer_name}):")
        for name, p in sorted(pp.items()):
            unit = _UNITS.get(name, "")
            unit_str = f" {unit}" if unit else ""
            lines.append(f"    {name} = {p['value']:.6g}{unit_str}  [{p['source_tag']}]")
    else:
        lines.append(f"  Polymer ({polymer_name}): (datasheet values only — nothing inferred)")

    # ── Datasheet (neat) values ────────────────────────────────────────────────
    lines.append("\nDatasheet (neat) properties:")
    if fiber:
        lines.append(f"  Fiber ({fiber_name}):")
        for k in ("neat_E1", "neat_E2", "neat_G12", "neat_nu12", "neat_nu23",
                  "neat_rho", "neat_k1", "neat_k2"):
            v = fiber.get(k)
            if v is not None:
                lines.append(f"    {k} = {v}")
    if polymer:
        lines.append(f"  Polymer ({polymer_name}):")
        for k in ("neat_E1", "neat_nu12", "neat_rho", "neat_k"):
            v = polymer.get(k)
            if v is not None:
                lines.append(f"    {k} = {v}")

    # ── Experimental measurements ──────────────────────────────────────────────
    exp = card.get("experimental_measurements", [])
    if exp:
        lines.append(f"\nExperimental measurements ({len(exp)} stored):")
        seen_exp: dict = {}
        for m in exp:
            n = m["property_name"]
            if n not in seen_exp:
                seen_exp[n] = m
        for n, m in sorted(seen_exp.items()):
            unc = f" ± {m['uncertainty']}" if m.get("uncertainty") else ""
            lines.append(f"  {n} = {m['value']}{unc} {m.get('unit','')}")

    return "\n".join(lines)


@tool
def get_material_details(material_name: str) -> str:
    """
    Get the complete datasheet properties for a single fiber or polymer by name
    (case-insensitive, partial match allowed — e.g. "AF", "carbon", "epoxy").
    Returns every stored property: moduli, Poisson ratios, density, CTE, conductivity.
    Call this when the user asks anything about a specific material — its properties,
    supplier, manufacturer, or any individual field like nu23 or k1.
    Do NOT use list_materials for single-material questions; use this tool instead.
    """
    name_lower = material_name.strip().lower()

    try:
        fibers   = _smat.list_fibers()
        polymers = _smat.list_polymers()
    except Exception as e:
        return f"Database error: {e}"

    matches = []

    for f in fibers:
        if name_lower in f["name"].lower():
            lines = [f"FIBER: {f['name']}  id={f['id']}  supplier={f.get('supplier','?')}"]
            _props = [
                ("E1",      f.get("neat_E1"),   "MPa"),
                ("E2",      f.get("neat_E2"),   "MPa"),
                ("G12",     f.get("neat_G12"),  "MPa"),
                ("nu12",    f.get("neat_nu12"), ""),
                ("nu23",    f.get("neat_nu23"), ""),
                ("density", f.get("neat_rho"),  "kg/m³"),
                ("CTE1",    f.get("neat_CTE1"), "1/K"),
                ("CTE2",    f.get("neat_CTE2"), "1/K"),
                ("k1",      f.get("neat_k1"),   "W/m·K"),
                ("k2",      f.get("neat_k2"),   "W/m·K"),
            ]
            for label, val, unit in _props:
                if val is not None:
                    lines.append(f"  {label} = {val} {unit}".rstrip())
            matches.append("\n".join(lines))

    for p in polymers:
        if name_lower in p["name"].lower():
            lines = [f"POLYMER: {p['name']}  id={p['id']}  supplier={p.get('supplier','?')}"]
            _props = [
                ("E (matrix modulus)", p.get("neat_E1"),   "MPa"),
                ("nu12",               p.get("neat_nu12"), ""),
                ("density",            p.get("neat_rho"),  "kg/m³"),
                ("k (conductivity)",   p.get("neat_k"),    "W/m·K"),
            ]
            for label, val, unit in _props:
                if val is not None:
                    lines.append(f"  {label} = {val} {unit}".rstrip())
            matches.append("\n".join(lines))

    if not matches:
        return f"No fiber or polymer found matching '{material_name}'. Call list_materials to see all available materials."

    return "\n\n".join(matches)


# ── Card inspection tool ──────────────────────────────────────────────────────

@tool
def inspect_card_inputs(card_id: int) -> str:
    """
    Show the exact inputs that would be passed to the forward model for a given
    material card — after full resolution (inferred values override datasheet,
    microstructure snapshot fills orientation fields).

    Use this BEFORE calling predict_properties so the user can see exactly what
    will be used and decide whether to override any values. Nothing is run or
    written — this is a read-only preview.

    Groups inputs into three sections:
      Fiber properties     — elastic constants and density from datasheet
      Polymer properties   — matrix modulus, Poisson ratio, density
                             (inferred value shown if Stage 1 is complete)
      Microstructure       — orientation tensor, fiber mass fraction, aspect ratio
                             (from Stage 1 snapshot if available)

    Also shows which CTE inputs are available (needed for thermoelastic prediction).
    """
    try:
        inputs = _scards.load_card_inputs(card_id)
        card   = _scards.load_card(card_id)
    except ValueError as e:
        return f"Card not found: {e}"
    except Exception as e:
        return f"Database error: {e}"

    cfg      = card["config"]
    fiber    = card.get("fiber")   or {}
    polymer  = card.get("polymer") or {}
    micro    = card.get("microstructure") or {}

    fname = fiber.get("name",   f"fiber_id={cfg['fiber_id']}")
    pname = polymer.get("name", f"polymer_id={cfg['polymer_id']}")

    # Determine provenance of key fields for display
    stages = _smat.get_completed_stages(card_id, cfg["fiber_id"], cfg["polymer_id"])

    lines = [
        f"CARD INPUTS PREVIEW — \"{cfg['name']}\"  (card_id={card_id})",
        f"  Fiber:   {fname}",
        f"  Polymer: {pname}",
        f"  Completed stages: {', '.join(stages) if stages else 'none'}",
        "",
        "These are the exact values that will be used in predict_properties.",
        "Override any field by passing it as an argument to predict_properties.",
        "",
    ]

    # ── Fiber properties ──────────────────────────────────────────────────────
    lines.append(f"FIBER PROPERTIES ({fname}):")
    _fiber_fields = [
        ("e1",           "axial modulus",      "MPa"),
        ("e2",           "transverse modulus", "MPa"),
        ("g12",          "shear modulus",      "MPa"),
        ("f_nu12",       "nu12",               ""),
        ("f_nu23",       "nu23",               ""),
        ("fiber_density","density",            "kg/m³"),
    ]
    for field, label, unit in _fiber_fields:
        v = inputs.get(field)
        if v is not None:
            unit_str = f" {unit}" if unit else ""
            lines.append(f"  {field:<18} = {v:.4g}{unit_str}   ({label})")

    # ── Polymer properties ────────────────────────────────────────────────────
    lines.append(f"\nPOLYMER PROPERTIES ({pname}):")
    em_tag  = "inferred — Stage 1" if "elastic" in stages else "datasheet"
    nu_tag  = "inferred — Stage 1" if "elastic" in stages else "datasheet"
    _poly_fields = [
        ("matrix_modulus", "matrix modulus", "MPa",  em_tag),
        ("matrix_poisson", "Poisson ratio",  "",     nu_tag),
        ("matrix_density", "density",        "kg/m³","datasheet"),
    ]
    for field, label, unit, tag in _poly_fields:
        v = inputs.get(field)
        if v is not None:
            unit_str = f" {unit}" if unit else ""
            lines.append(f"  {field:<18} = {v:.4g}{unit_str}   ({label}, {tag})")

    # ── Microstructure ────────────────────────────────────────────────────────
    lines.append("\nMICROSTRUCTURE:")
    micro_tag = "Stage 1 snapshot" if "elastic" in stages else "NOT AVAILABLE"
    _micro_fields = [
        ("a11",           "alignment — print direction"),
        ("a22",           "alignment — transverse"),
        ("a12",           "shear xy"),
        ("a13",           "shear xz"),
        ("a23",           "shear yz"),
        ("fiber_massfrac","fiber mass fraction"),
        ("ar",            "aspect ratio"),
    ]
    for field, label in _micro_fields:
        v = inputs.get(field)
        if v is not None:
            lines.append(f"  {field:<18} = {v:.4g}   ({label}, {micro_tag})")
        else:
            lines.append(f"  {field:<18}   MISSING — provide as override in predict_properties")

    # ── CTE inputs (thermoelastic) ────────────────────────────────────────────
    lines.append("\nCTE INPUTS (needed for thermoelastic prediction):")
    cte_tag = "inferred — Stage 2" if "thermoelastic" in stages else "NOT AVAILABLE"
    _cte_fields = [
        ("f_cte1", "fiber axial CTE",     "1/K"),
        ("f_cte2", "fiber transverse CTE","1/K"),
        ("m_cte",  "matrix CTE",          "1/K"),
    ]
    for field, label, unit in _cte_fields:
        v = inputs.get(field)
        if v is not None:
            lines.append(f"  {field:<18} = {v:.4e} {unit}   ({v*1e6:.3f} ppm/K, {cte_tag})")
        else:
            lines.append(f"  {field:<18}   MISSING — run Stage 2 or provide as override")

    # ── Readiness summary ─────────────────────────────────────────────────────
    elastic_missing = [f for f in _ELASTIC_REQUIRED if f not in inputs]
    te_missing      = [f for f in _TE_EXTRA_REQUIRED if f not in inputs]

    lines.append("\nREADINESS:")
    if not elastic_missing:
        lines.append("  Elastic prediction     — READY")
    else:
        lines.append(f"  Elastic prediction     — NOT READY (missing: {elastic_missing})")
    if not te_missing:
        lines.append("  Thermoelastic prediction — READY")
    else:
        lines.append(f"  Thermoelastic prediction — NOT READY (missing: {te_missing})")

    return "\n".join(lines)


# ── Forward prediction tools ──────────────────────────────────────────────────

@tool
def get_model_inputs_outputs(model_name: str) -> str:
    """
    Return the exact input fields required and output fields produced by a forward
    surrogate model. Call this before predict_properties to verify what a model needs.
    model_name must be "elastic" or "thermoelastic".
    """
    try:
        inputs  = _sfwd.get_input_fields(model_name)
        outputs = _sfwd.get_output_fields(model_name)
    except FileNotFoundError:
        return f"Model '{model_name}' not found. Use 'elastic' or 'thermoelastic'."
    except Exception as e:
        return f"Error loading model: {e}"

    _INPUT_DESCS = {
        "e1": "fiber axial modulus (MPa)",
        "e2": "fiber transverse modulus (MPa)",
        "g12": "fiber shear modulus (MPa)",
        "f_nu12": "fiber Poisson ratio nu12",
        "f_nu23": "fiber Poisson ratio nu23",
        "ar": "fiber aspect ratio",
        "fiber_massfrac": "fiber mass fraction (0–1)",
        "fiber_density": "fiber density (kg/m³)",
        "matrix_modulus": "matrix Young's modulus (MPa)",
        "matrix_poisson": "matrix Poisson ratio",
        "matrix_density": "matrix density (kg/m³)",
        "a11": "orientation tensor — print direction",
        "a22": "orientation tensor — transverse",
        "a12": "orientation tensor — shear xy",
        "a13": "orientation tensor — shear xz",
        "a23": "orientation tensor — shear yz",
        "f_cte1": "fiber axial CTE (1/K)",
        "f_cte2": "fiber transverse CTE (1/K)",
        "m_cte": "matrix CTE (1/K)",
    }
    _OUTPUT_DESCS = {
        "E1": "axial modulus (MPa)", "E2": "transverse modulus (MPa)",
        "E3": "out-of-plane modulus (MPa)",
        "G12": "in-plane shear (MPa)", "G13": "shear (MPa)", "G23": "shear (MPa)",
        "nu12": "Poisson ratio", "nu13": "Poisson ratio", "nu23": "Poisson ratio",
        "CTE11": "axial CTE (1/K)", "CTE22": "transverse CTE (1/K)",
        "CTE33": "out-of-plane CTE (1/K)",
        "CTE12": "CTE shear coupling", "CTE13": "CTE shear coupling",
        "CTE23": "CTE shear coupling",
    }

    lines = [f"MODEL: {model_name.upper()}", "", "INPUTS (all required):"]
    for f in inputs:
        lines.append(f"  {f:<20} {_INPUT_DESCS.get(f, '')}")
    lines.append("\nOUTPUTS:")
    for f in outputs:
        lines.append(f"  {f:<20} {_OUTPUT_DESCS.get(f, '')}")
    return "\n".join(lines)


@tool
def predict_properties(
    card_id: int = -1,
    fiber_id: int = -1,
    polymer_id: int = -1,
    a11: float = -1.0,
    a22: float = -1.0,
    a12: float = -1.0,
    a13: float = -1.0,
    a23: float = -1.0,
    fiber_massfrac: float = -1.0,
    ar: float = -1.0,
    matrix_modulus_MPa: float = -1.0,
    matrix_poisson: float = -1.0,
    f_cte1_per_K: float = -1.0,
    f_cte2_per_K: float = -1.0,
    m_cte_per_K: float = -1.0,
) -> str:
    """
    Run elastic (and optionally thermoelastic) forward prediction.

    Three ways to specify the material system — pick ONE:
      1. card_id >= 0: load everything from a saved material card (fiber + polymer
         + microstructure + any inferred constituent properties). Best option when
         Stage 1 has already been run.
      2. fiber_id >= 0 AND polymer_id >= 0 (no card): load fiber and polymer
         datasheet properties. You MUST then provide all microstructure fields
         explicitly via the override arguments (a11, a22, a12, a13, a23,
         fiber_massfrac, ar).

    Any argument set to a value != -1.0 overrides what was loaded from the card or
    datasheet. Use this for what-if scenarios — e.g. change a11 while keeping
    everything else from a saved card.

    Thermoelastic prediction (CTE11, CTE22, CTE33) runs automatically when
    f_cte1_per_K, f_cte2_per_K, and m_cte_per_K are all available — either
    from a card with Stage 2 complete or from explicit override arguments.

    Units: moduli in MPa, CTE in 1/K (NOT ppm/K — multiply ppm/K by 1e-6 first).
    """
    # ── Step 1: Load base inputs ──────────────────────────────────────────────
    inputs = {}
    source_label = ""
    overrides_applied = []

    if card_id >= 0:
        try:
            card   = _scards.load_card(card_id)
            inputs = _scards.load_card_inputs(card_id)
            cfg    = card["config"]
            fname  = (card.get("fiber")   or {}).get("name", f"fiber_id={cfg['fiber_id']}")
            pname  = (card.get("polymer") or {}).get("name", f"polymer_id={cfg['polymer_id']}")
            source_label = f"Card #{card_id} \"{cfg['name']}\"  ({fname} / {pname})"
        except ValueError as e:
            return f"Card not found: {e}"
        except Exception as e:
            return f"Database error loading card: {e}"

    elif fiber_id >= 0 and polymer_id >= 0:
        try:
            inputs   = _smat.get_model_inputs(fiber_id, polymer_id, use_inferred=True)
            fibers   = {f["id"]: f["name"] for f in _smat.list_fibers()}
            polymers = {p["id"]: p["name"] for p in _smat.list_polymers()}
            fname    = fibers.get(fiber_id,   f"id={fiber_id}")
            pname    = polymers.get(polymer_id, f"id={polymer_id}")
            source_label = f"Datasheets: {fname} / {pname}"
        except Exception as e:
            return f"Database error loading materials: {e}"

    else:
        return (
            "Specify the material system:\n"
            "  option A — card_id >= 0  (loads fiber + polymer + microstructure from a saved card)\n"
            "  option B — fiber_id >= 0 AND polymer_id >= 0  "
            "(loads datasheets; you must also provide a11, a22, a12, a13, a23, fiber_massfrac, ar)"
        )

    # ── Step 2: Apply overrides (sentinel -1.0 means "not provided") ─────────
    _override_map = [
        ("a11",            a11),
        ("a22",            a22),
        ("a12",            a12),
        ("a13",            a13),
        ("a23",            a23),
        ("fiber_massfrac", fiber_massfrac),
        ("ar",             ar),
        ("matrix_modulus", matrix_modulus_MPa),
        ("matrix_poisson", matrix_poisson),
        ("f_cte1",         f_cte1_per_K),
        ("f_cte2",         f_cte2_per_K),
        ("m_cte",          m_cte_per_K),
    ]
    for field, val in _override_map:
        if val != -1.0:
            prev = inputs.get(field)
            inputs[field] = val
            if prev is not None and abs(float(prev) - val) > 1e-12:
                overrides_applied.append(
                    f"  {field}: {float(prev):.4g} → {val:.4g}"
                )
            else:
                overrides_applied.append(f"  {field} = {val:.4g}  [provided]")

    # ── Orientation tensor semi-positive definiteness check ──────────────────
    _a11 = inputs.get("a11")
    _a22 = inputs.get("a22")
    if _a11 is not None and _a22 is not None:
        _a33 = 1.0 - _a11 - _a22
        if _a33 < 0.0:
            return (
                f"Invalid orientation tensor: a11 + a22 = {_a11:.4g} + {_a22:.4g} = {_a11 + _a22:.4g} > 1.\n"
                f"a33 = 1 - a11 - a22 = {_a33:.4g} < 0, which violates semi-positive definiteness.\n"
                f"Reduce a11 or a22 so their sum is at most 1.0."
            )

    # Ensure both density aliases are present
    if "matrix_density" not in inputs and "rho_m" in inputs:
        inputs["matrix_density"] = inputs["rho_m"]
    if "rho_m" not in inputs and "matrix_density" in inputs:
        inputs["rho_m"] = inputs["matrix_density"]

    # ── Step 3: Check what can run ────────────────────────────────────────────
    elastic_missing = [f for f in _ELASTIC_REQUIRED if f not in inputs]
    te_missing      = [f for f in _TE_EXTRA_REQUIRED if f not in inputs]

    # ── Step 4: Build report header ───────────────────────────────────────────
    lines = [f"FORWARD PREDICTION\nSource: {source_label}"]

    if overrides_applied:
        lines.append("\nOverrides applied:")
        lines.extend(overrides_applied)

    # Show the microstructure and key constituent values that will be used
    lines.append("\nInputs used:")
    micro_fields = ["a11", "a22", "a12", "a13", "a23", "fiber_massfrac", "ar"]
    micro_vals = [(f, inputs[f]) for f in micro_fields if f in inputs]
    if micro_vals:
        lines.append("  Microstructure:")
        for f, v in micro_vals:
            lines.append(f"    {f:<18} = {v:.4g}")
    em  = inputs.get("matrix_modulus")
    nu  = inputs.get("matrix_poisson")
    if em is not None:
        lines.append(f"  matrix_modulus       = {em:.1f} MPa")
    if nu is not None:
        lines.append(f"  matrix_poisson       = {nu:.4f}")
    if "f_cte1" in inputs:
        lines.append(f"  f_cte1               = {inputs['f_cte1']:.3e} /K")
    if "f_cte2" in inputs:
        lines.append(f"  f_cte2               = {inputs['f_cte2']:.3e} /K")
    if "m_cte" in inputs:
        lines.append(f"  m_cte                = {inputs['m_cte']:.3e} /K")

    # ── Step 5: Run elastic model ─────────────────────────────────────────────
    if elastic_missing:
        lines.append(f"\nCANNOT RUN: missing required fields: {elastic_missing}")
        if card_id >= 0:
            lines.append(
                "  Stage 1 (elastic inverse) may not have been run for this card yet.\n"
                "  Run Stage 1 first, or provide microstructure overrides directly."
            )
        else:
            lines.append(
                "  Provide all microstructure fields: a11, a22, a12, a13, a23, fiber_massfrac, ar"
            )
        return "\n".join(lines)

    try:
        el_out = _sfwd.run_forward("elastic", inputs)
    except Exception as e:
        return "\n".join(lines) + f"\n\nElastic model error: {e}"

    lines.append("\nELASTIC PREDICTIONS:")
    for name in _sfwd.get_output_fields("elastic"):
        v = el_out.get(name)
        if v is None:
            continue
        if name.startswith("E") or name.startswith("G"):
            lines.append(f"  {name:<6} = {v:>10.1f} MPa   ({v/1000:.3f} GPa)")
        else:
            lines.append(f"  {name:<6} = {v:>10.5f}")

    # ── Step 6: Run thermoelastic model if CTE inputs present ────────────────
    if te_missing:
        lines.append(f"\nTHERMOELASTIC SKIPPED: missing {te_missing}")
        lines.append(
            "  Provide f_cte1_per_K, f_cte2_per_K, m_cte_per_K as overrides,\n"
            "  or run Stage 2 (thermoelastic inverse) to infer them and save to card."
        )
    else:
        try:
            te_out = _sfwd.run_forward("thermoelastic", inputs)
        except Exception as e:
            lines.append(f"\nThermoelastic model error: {e}")
            return "\n".join(lines)

        lines.append("\nTHERMOELASTIC PREDICTIONS:")
        for name in _sfwd.get_output_fields("thermoelastic"):
            if not name.startswith("CTE"):
                continue
            v = te_out.get(name)
            if v is None:
                continue
            lines.append(f"  {name:<8} = {v:.4e} /K   ({v * 1e6:.3f} ppm/K)")

    return "\n".join(lines)


# ── Material library tools ────────────────────────────────────────────────────

@tool
def add_fiber(
    name: str,
    supplier: str,
    E1_MPa: float,
    E2_MPa: float,
    G12_MPa: float,
    nu12: float,
    nu23: float,
    density_kg_m3: float,
) -> str:
    """
    Add a new fiber to the material library.

    Required mechanical datasheet properties (all from the fiber datasheet):
      E1_MPa, E2_MPa, G12_MPa — moduli in MPa  (if given in GPa, multiply by 1000)
      nu12, nu23               — Poisson ratios (dimensionless, typically 0.1–0.4)
      density_kg_m3            — density in kg/m³  (e.g. carbon fiber ≈ 1750–1800)

    CTE and thermal conductivity are NOT stored here — they are inferred later from
    composite measurements during Stage 2 (CTE) and Stage 3 (thermal conductivity).

    Returns the new fiber id to use in subsequent calls (fiber_id parameter).
    """
    try:
        fid = _smat.add_fiber(
            name=name.strip(),
            supplier=supplier.strip(),
            E1=E1_MPa, E2=E2_MPa, G12=G12_MPa,
            nu12=nu12, nu23=nu23, rho=density_kg_m3,
        )
    except ValueError as e:
        return f"Cannot add fiber: {e}"
    except Exception as e:
        return f"Database error: {e}"

    return (
        f"Fiber added successfully.\n"
        f"  fiber_id = {fid}\n"
        f"  name     = \"{name}\"\n"
        f"  supplier = \"{supplier}\"\n"
        f"  E1 = {E1_MPa} MPa,  E2 = {E2_MPa} MPa,  G12 = {G12_MPa} MPa\n"
        f"  nu12 = {nu12},  nu23 = {nu23},  density = {density_kg_m3} kg/m³\n"
        f"Use fiber_id={fid} in list_materials, get_material_details, and predict_properties."
    )


@tool
def add_polymer(
    name: str,
    supplier: str,
    E_MPa: float,
    nu12: float,
    density_kg_m3: float,
) -> str:
    """
    Add a new polymer (matrix material) to the material library.

    Required datasheet properties:
      E_MPa         — Young's modulus (matrix modulus) in MPa
      nu12          — Poisson ratio (typically 0.33–0.42 for thermoplastics)
      density_kg_m3 — density in kg/m³  (e.g. PESU ≈ 1280, epoxy ≈ 1200)

    Returns the new polymer id to use in subsequent calls (polymer_id parameter).
    """
    try:
        pid = _smat.add_polymer(
            name=name.strip(),
            supplier=supplier.strip(),
            E1=E_MPa,
            nu12=nu12,
            rho=density_kg_m3,
        )
    except ValueError as e:
        return f"Cannot add polymer: {e}"
    except Exception as e:
        return f"Database error: {e}"

    return (
        f"Polymer added successfully.\n"
        f"  polymer_id = {pid}\n"
        f"  name       = \"{name}\"\n"
        f"  supplier   = \"{supplier}\"\n"
        f"  E = {E_MPa} MPa,  nu12 = {nu12},  density = {density_kg_m3} kg/m³\n"
        f"Use polymer_id={pid} in list_materials, get_material_details, and predict_properties."
    )


# ── Identifiability tool ──────────────────────────────────────────────────────

@tool
def check_identifiability(
    free_variables: str,
    available_measurements: str,
    fiber_id: int = -1,
    polymer_id: int = -1,
    card_id: int = -1,
) -> str:
    """
    Fisher Information Matrix (FIM) identifiability analysis.

    Answers questions like: "Can I infer a11 and a22 if I only have E1?"
    Returns a per-parameter verdict (WELL / MARGINAL / POOR) and recommends
    which additional measurements would most improve identifiability.

    free_variables: comma-separated list of parameters to infer.
      Examples: "a11,a22"  |  "a11 a22 fiber_massfrac"  |  "matrix_modulus matrix_poisson"
      Supported: a11 a22 a12 a13 a23, fiber_massfrac (or mf/wf), ar, matrix_modulus,
                 matrix_poisson, f_cte1, f_cte2, m_cte

    available_measurements: comma-separated list of composite outputs you have measured.
      Examples: "E1"  |  "E1,E2,G12,nu12"  |  "CTE11,CTE22"
      Supported elastic: E1 E2 E3 G12 G13 G23 nu12 nu13 nu23
      Supported thermoelastic: CTE11 CTE22 CTE33

    fiber_id, polymer_id, card_id: optional — provide realistic nominal material
      properties for the analysis. If none are given, generic carbon/polymer
      defaults are used (still gives correct structural identifiability result).

    The model is selected automatically: elastic for mechanical unknowns/measurements,
    thermoelastic if CTE unknowns (f_cte1, f_cte2, m_cte) or CTE measurements are present.
    """
    # ── Parse free_variables ──────────────────────────────────────────────────
    raw_free = [t.strip().lower() for t in re.split(r"[,\s]+", free_variables.strip()) if t.strip()]
    free_inputs = []
    unrecognized_free = []
    for tok in raw_free:
        canonical = _FREE_SYNONYMS.get(tok)
        if canonical and canonical not in free_inputs:
            free_inputs.append(canonical)
        elif not canonical:
            unrecognized_free.append(tok)

    if not free_inputs:
        return (
            f"Could not parse any recognized free variables from: '{free_variables}'\n"
            f"Examples: 'a11,a22'  |  'a11 a22 fiber_massfrac'  |  'matrix_modulus'\n"
            f"Unrecognized: {unrecognized_free}"
        )

    # ── Parse available_measurements ──────────────────────────────────────────
    raw_meas = [t.strip() for t in re.split(r"[,\s]+", available_measurements.strip()) if t.strip()]
    meas_inputs = []
    unrecognized_meas = []
    for tok in raw_meas:
        canonical = (
            _MEAS_SYNONYMS.get(tok)
            or _MEAS_SYNONYMS.get(tok.upper())
            or _MEAS_SYNONYMS.get(tok.lower())
        )
        if canonical and canonical not in meas_inputs:
            meas_inputs.append(canonical)
        elif not canonical:
            unrecognized_meas.append(tok)

    if not meas_inputs:
        return (
            f"Could not parse any recognized measurements from: '{available_measurements}'\n"
            f"Elastic outputs: E1 E2 E3 G12 G13 G23 nu12 nu13 nu23\n"
            f"Thermoelastic:   CTE11 CTE22 CTE33\n"
            f"Unrecognized: {unrecognized_meas}"
        )

    # ── Select model ──────────────────────────────────────────────────────────
    _TE_FREE = {"f_cte1", "f_cte2", "m_cte"}
    _TE_MEAS = {"CTE11", "CTE22", "CTE33", "CTE12", "CTE13", "CTE23"}
    model_name = (
        "thermoelastic"
        if (set(free_inputs) & _TE_FREE) or (set(meas_inputs) & _TE_MEAS)
        else "elastic"
    )

    # ── Validate bounds ───────────────────────────────────────────────────────
    missing_bounds = [v for v in free_inputs if v not in _DEFAULT_BOUNDS]
    if missing_bounds:
        return f"No default bounds defined for: {missing_bounds}. Cannot run FIM."
    bounds = {v: _DEFAULT_BOUNDS[v] for v in free_inputs}

    # ── Build nominal inputs ──────────────────────────────────────────────────
    if card_id >= 0:
        try:
            nominal = _scards.load_card_inputs(card_id)
            card    = _scards.load_card(card_id)
            cfg     = card["config"]
            context_label = f"Card #{card_id} \"{cfg['name']}\""
        except Exception as e:
            return f"Error loading card: {e}"
    elif fiber_id >= 0 and polymer_id >= 0:
        try:
            nominal  = _smat.get_model_inputs(fiber_id, polymer_id, use_inferred=True)
            fibers   = {f["id"]: f["name"] for f in _smat.list_fibers()}
            polymers = {p["id"]: p["name"] for p in _smat.list_polymers()}
            context_label = (
                f"{fibers.get(fiber_id, f'fiber {fiber_id}')} / "
                f"{polymers.get(polymer_id, f'polymer {polymer_id}')}"
            )
        except Exception as e:
            return f"Error loading materials: {e}"
    else:
        nominal = dict(_NOMINAL_INPUTS)
        context_label = "generic nominal values (carbon/polymer defaults)"

    # Fill any gaps from _NOMINAL_INPUTS so the model has a complete input vector
    for k, v in _NOMINAL_INPUTS.items():
        if k not in nominal:
            nominal[k] = v

    # ── Validate measurements against model outputs ───────────────────────────
    try:
        valid_outputs = set(_sfwd.get_output_fields(model_name))
    except Exception as e:
        return f"Error loading {model_name} model: {e}"

    invalid_meas = [m for m in meas_inputs if m not in valid_outputs]
    if invalid_meas:
        return (
            f"These measurements are not outputs of the {model_name} model: {invalid_meas}\n"
            f"Valid {model_name} outputs: {sorted(valid_outputs)}"
        )

    # ── Run FIM ───────────────────────────────────────────────────────────────
    target_outputs = {m: 0.0 for m in meas_inputs}
    try:
        result = _sfim.run_fim(
            model_name=model_name,
            fixed_inputs=nominal,
            free_inputs=free_inputs,
            bounds=bounds,
            target_outputs=target_outputs,
            n_samples=200,
        )
    except Exception as e:
        return f"FIM analysis error: {e}"

    status  = result["status"]
    cond    = result["condition_number"]
    recs    = result["recommendations"]
    cr_std  = result["cramer_rao_std"]

    _STATUS_LABEL = {"WELL": "WELL", "MARGINAL": "MARGINAL", "POOR": "POOR"}

    # ── Format results ────────────────────────────────────────────────────────
    lines = [
        f"IDENTIFIABILITY ANALYSIS — {model_name.upper()} MODEL",
        f"Free variables (want to infer): {', '.join(free_inputs)}",
        f"Available measurements:         {', '.join(meas_inputs)}",
        f"Context: {context_label}",
        "",
        "RESULT PER PARAMETER:",
    ]

    for param in free_inputs:
        s  = status.get(param, "POOR")
        cr = cr_std.get(param)
        if cr is not None and s != "POOR":
            if "cte" in param:
                cr_info = f"   Cramér-Rao std ≈ {cr:.2e} /K  ({cr * 1e6:.2f} ppm/K)"
            elif "modulus" in param:
                cr_info = f"   Cramér-Rao std ≈ {cr:.1f} MPa"
            else:
                cr_info = f"   Cramér-Rao std ≈ {cr:.4f}"
        else:
            cr_info = ""
        lines.append(f"  {param:<20}  {_STATUS_LABEL[s]}{cr_info}")

    cond_str = "inf (singular)" if (cond == float("inf") or cond > 1e12) else f"{cond:.1f}"
    lines.append(f"\nCondition number: {cond_str}")

    # Plain-English verdict
    all_poor  = all(v == "POOR"     for v in status.values())
    any_poor  = any(v == "POOR"     for v in status.values())
    any_marg  = any(v == "MARGINAL" for v in status.values())

    n_free = len(free_inputs)
    n_meas = len(meas_inputs)

    if all_poor:
        lines.append(
            f"\nVERDICT: UNDERDETERMINED\n"
            f"  {n_meas} measurement(s) is not enough to identify {n_free} unknown(s).\n"
            f"  The available measurements have too little sensitivity to these parameters."
        )
    elif any_poor:
        poor = [p for p in free_inputs if status.get(p) == "POOR"]
        lines.append(
            f"\nVERDICT: PARTIALLY IDENTIFIABLE\n"
            f"  These parameters cannot be reliably inferred: {poor}\n"
            f"  Add more measurements (see recommendations below)."
        )
    elif any_marg:
        marg = [p for p in free_inputs if status.get(p) == "MARGINAL"]
        lines.append(
            f"\nVERDICT: MARGINALLY IDENTIFIABLE\n"
            f"  All parameters are borderline: {marg}\n"
            f"  Results will carry significant uncertainty. More measurements would help."
        )
    else:
        lines.append(
            f"\nVERDICT: WELL DETERMINED\n"
            f"  {n_meas} measurement(s) can reliably identify all {n_free} parameter(s)."
        )

    # Recommendations
    if recs:
        lines.append("\nRECOMMENDED ADDITIONAL MEASUREMENTS (ranked by improvement):")
        for i, r in enumerate(recs[:5], 1):
            eig = r["new_min_eigenvalue"]
            if eig >= 100:
                benefit = "→ WELL determined"
            elif eig >= 1:
                benefit = "→ MARGINAL"
            else:
                benefit = f"→ min eigenvalue {eig:.3f} (still POOR)"
            lines.append(f"  {i}. {r['name']:<8}  {benefit}")

    if unrecognized_free:
        lines.append(f"\n(Unrecognized free variable terms ignored: {unrecognized_free})")
    if unrecognized_meas:
        lines.append(f"(Unrecognized measurement terms ignored: {unrecognized_meas})")

    return "\n".join(lines)


# ── All tools passed to the LLM and ToolNode ─────────────────────────────────

TOOLS = [
    search_knowledge_base,
    list_materials,
    get_material_details,
    list_cards,
    get_card_status,
    inspect_card_inputs,
    get_model_inputs_outputs,
    predict_properties,
    add_fiber,
    add_polymer,
    check_identifiability,
]
