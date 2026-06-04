"""
All tools available to the DiffMicromechanics agent.

Each tool is a @tool-decorated function. Add new tools here as capabilities
are built out (elastic inverse, save to card, etc.).

Current tools:
    search_knowledge_base         — RAG over agent/knowledge/ PDFs
    list_materials                — list all fibers, polymers, printers in the DB
    get_material_details          — full datasheet for one fiber or polymer by name
    convert_fraction              — convert fiber mass fraction ↔ volume fraction using DB densities
    list_cards                    — list all material cards and their stage status
    get_card_status               — full detail view of one material card
    get_model_inputs_outputs      — field names + units for elastic/thermoelastic models
    inspect_card_inputs           — preview resolved inputs before predicting
    predict_properties            — forward prediction (elastic + thermoelastic)
    predict_thermal_conductivity  — forward prediction (thermal) at one T or across a T range
    add_fiber                     — add a new fiber to the material library
    add_polymer                   — add a new polymer to the material library
    check_identifiability         — FIM analysis: can these measurements identify these unknowns?
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Must be set before any JAX import (forward.py imports JAX at module level)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

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
import core.services.service_inverse as _sinv
import core.inverse_thermal as _ithermal


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

_THERMAL_STRUCTURAL_REQUIRED = frozenset([
    "ar_f", "w_f", "rho_f", "rho_m",
    "a11", "a22", "a12", "a13", "a23",
])

_THERMAL_DEFAULT_TEMPS = [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0]

# ── Stage 1 elastic inverse constants ────────────────────────────────────────

# Fields optimised in Stage 1 — never change per-call
_STAGE1_FREE = [
    "a11", "a22",
    "fiber_massfrac", "ar",
    "matrix_modulus", "matrix_poisson",
]

# matrix_poisson is only added to free list when shear/Poisson measurements are present
# (G12, G13, G23, nu12, nu13, nu23) — it is not identifiable from E1/E2/E3 alone.
# a12, a13, a23 are not in the free list — they default to 0.0 (fixed).
# ar and fiber_massfrac are in the list so they CAN be freed, but the datasheet
# fallback below ensures they default to fixed unless the user explicitly requests inference.
_STAGE1_FREE_WITHOUT_POISSON = [
    "a11", "a22",
    "fiber_massfrac", "ar",
    "matrix_modulus",
]
_SHEAR_POISSON_MEASUREMENTS = frozenset(["G12", "G13", "G23", "nu12", "nu13", "nu23"])

# Keys to exclude from fixed_inputs (free vars + their field-name aliases)
_STAGE1_FREE_KEYS = frozenset([
    "a11", "a22", "a12", "a13", "a23",
    "fiber_massfrac", "w_f",
    "ar", "ar_f",
    "matrix_modulus", "matrix_poisson",
])

# Holds the last solver result between a run_* call and save_to_card
# Structure: {"result": dict compatible with save_inverse_result, "meta": {fiber_id, ...}}
_pending_save: dict = {}

_T_REF = 1.0  # reference temperature for polymer conductivity model (°C), matches inverse_thermal.py


def _resolve_k_at_T(inputs: dict, T: float) -> tuple:
    """Return (k_f1, k_f2, k_m) at temperature T.

    Resolution order:
      1. Stage 3 parametric model: p1 + p2 (polymer) + k_f1 + k_f2 (fiber, constant).
         The DB stores derived k_f1=l2 and k_f2=l2/t directly — not l2/t themselves.
         k_m(T) = p1 * sqrt(T / T_ref) + p2   (temperature-dependent)
         k_f1, k_f2 are constant (temperature-independent).
      2. Scalar k_f1, k_f2, k_m directly in inputs (datasheet or previously stored).
    Returns (None, None, None) if no thermal data is available.
    """
    p1  = inputs.get("p1")
    p2  = inputs.get("p2")
    k_f1 = inputs.get("k_f1")
    k_f2 = inputs.get("k_f2")
    if all(v is not None for v in (p1, p2, k_f1, k_f2)):
        k_m = float(p1) * (max(T, 0.0) / _T_REF) ** 0.5 + float(p2)
        return float(k_f1), float(k_f2), k_m

    k_m = inputs.get("k_m")
    if all(v is not None for v in (k_f1, k_f2, k_m)):
        return float(k_f1), float(k_f2), float(k_m)

    return None, None, None


_DEFAULT_BOUNDS = {
    "a11":            (0.50,    0.85),
    "a22":            (0.01,    0.4),   # matches problem.json
    "a12":            (-0.10,   0.10),
    "a13":            (-0.10,   0.10),
    "a23":            (-0.10,   0.10),
    "fiber_massfrac": (0.05,    0.60),
    "ar":             (5.0,    100.0),
    "matrix_modulus": (2000.0, 5000.0),  # matches problem.json
    "matrix_poisson": (0.33,    0.42),   # matches problem.json
    "f_cte1":         (-2e-6,   5e-6),
    "f_cte2":         (5e-6,   30e-6),
    "m_cte":          (30e-6, 120e-6),
}

# Solver config that matches the GUI's problem.json defaults
_ELASTIC_SOLVER_CFG = {
    "method":             "lbfgsb",
    "constraint_penalty": 10000.0,
    "use_epsilon_loss":   False,
    "epsilon_scale":      0.5,
    "maxiter":            300,
    "tol":                1e-6,
    "seed":               42,
}

# Stage 2 — thermoelastic free variables and initial guesses
_TE_FREE = ["f_cte1", "f_cte2", "m_cte"]
_TE_INIT = {
    "f_cte1":  1.5e-6,   # midpoint of bounds; carbon ~0, glass ~5e-6
    "f_cte2": 17.5e-6,   # midpoint of bounds
    "m_cte":  75.0e-6,   # midpoint of bounds; polymers typically 50-100 ppm/K
}

# Stage 3 — thermal inverse forward model (lazy-loaded on first call)
_THERMAL_FWD_MODEL = None

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

    # ── Processing conditions ──────────────────────────────────────────────────
    pc = card.get("processing_condition")
    lines.append("\nPrinting conditions:")
    if pc:
        if pc.get("bead_width")      is not None: lines.append(f"  bead_width      = {pc['bead_width']} mm")
        if pc.get("bead_height")     is not None: lines.append(f"  bead_height     = {pc['bead_height']} mm")
        if pc.get("nozzle_diameter") is not None: lines.append(f"  nozzle_diameter = {pc['nozzle_diameter']} mm")
        if pc.get("speed")           is not None: lines.append(f"  print_speed     = {pc['speed']} mm/s")
        if pc.get("notes"):                       lines.append(f"  notes           = {pc['notes']}")
    else:
        lines.append("  (none recorded — use save_processing_conditions to add)")

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


@tool
def convert_fraction(
    fiber_name: str,
    polymer_name: str,
    fiber_massfrac: float = -1.0,
    fiber_volfrac: float = -1.0,
) -> str:
    """
    Convert between fiber mass fraction (wf) and fiber volume fraction (Vf)
    using material densities from the database.

    Pass exactly one of:
      fiber_massfrac — convert wf → Vf
      fiber_volfrac  — convert Vf → wf

    Use this tool for any fraction conversion. Do NOT compute manually.
    """
    if fiber_massfrac < 0 and fiber_volfrac < 0:
        return "Provide either fiber_massfrac or fiber_volfrac (not both negative)."
    if fiber_massfrac >= 0 and fiber_volfrac >= 0:
        return "Provide only one of fiber_massfrac or fiber_volfrac, not both."

    try:
        fiber_id, polymer_id = _resolve_fiber_polymer(fiber_name, polymer_name)
    except ValueError as e:
        return str(e)

    try:
        if fiber_massfrac >= 0:
            result = _smat.convert_mass_to_volume_fraction(
                fiber_id=fiber_id, polymer_id=polymer_id,
                mass_fraction=fiber_massfrac,
            )
            return (
                f"Fiber: {fiber_name}  ρ_f = {result['rho_f']} kg/m³\n"
                f"Polymer: {polymer_name}  ρ_m = {result['rho_m']} kg/m³\n"
                f"Mass fraction wf = {result['wf']}\n"
                f"Volume fraction Vf = {result['vf']:.4f}"
            )
        else:
            result = _smat.convert_volume_to_mass_fraction(
                fiber_id=fiber_id, polymer_id=polymer_id,
                volume_fraction=fiber_volfrac,
            )
            return (
                f"Fiber: {fiber_name}  ρ_f = {result['rho_f']} kg/m³\n"
                f"Polymer: {polymer_name}  ρ_m = {result['rho_m']} kg/m³\n"
                f"Volume fraction Vf = {result['vf']}\n"
                f"Mass fraction wf = {result['wf']:.4f}"
            )
    except ValueError as e:
        return f"Cannot compute: {e}"


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

    # ── Thermal conductivity inputs ───────────────────────────────────────────
    lines.append("\nTHERMAL CONDUCTIVITY INPUTS:")
    has_parametric = all(inputs.get(p) is not None for p in ("p1", "p2", "k_f1", "k_f2"))
    has_scalar_k   = all(inputs.get(k) is not None for k in ("k_f1", "k_f2", "k_m"))

    if has_parametric:
        lines.append("  Stage 3 parametric model available (temperature-dependent):")
        for field, label in [("p1","polymer k scaling"), ("p2","polymer k offset"),
                              ("l2","fiber k_f1"), ("t","fiber anisotropy k_f1/k_f2")]:
            v = inputs.get(field)
            if v is not None:
                lines.append(f"  {field:<6} = {v:.4e} W/m·K   ({label})")
    elif has_scalar_k:
        lines.append("  Scalar k values available (temperature-independent):")
        for field, label in [("k_f1","fiber axial k"), ("k_f2","fiber transverse k"),
                              ("k_m","matrix k")]:
            v = inputs.get(field)
            lines.append(f"  {field:<6} = {v:.5f} W/m·K   ({label})")
    else:
        lines.append("  NOT AVAILABLE — run Stage 3 (thermal inverse) or provide k_f1_WmK,")
        lines.append("  k_f2_WmK, k_m_WmK as overrides in predict_thermal_conductivity.")

    # ── Readiness summary ─────────────────────────────────────────────────────
    elastic_missing  = [f for f in _ELASTIC_REQUIRED if f not in inputs]
    te_missing       = [f for f in _TE_EXTRA_REQUIRED if f not in inputs]
    struct_missing   = [f for f in _THERMAL_STRUCTURAL_REQUIRED if f not in inputs]
    thermal_k_ready  = has_parametric or has_scalar_k

    lines.append("\nREADINESS:")
    if not elastic_missing:
        lines.append("  Elastic prediction          — READY")
    else:
        lines.append(f"  Elastic prediction          — NOT READY (missing: {elastic_missing})")
    if not te_missing:
        lines.append("  Thermoelastic prediction    — READY")
    else:
        lines.append(f"  Thermoelastic prediction    — NOT READY (missing: {te_missing})")
    if not struct_missing and thermal_k_ready:
        k_mode = "temperature-dependent" if has_parametric else "scalar (temperature-independent)"
        lines.append(f"  Thermal conductivity        — READY ({k_mode})")
    elif struct_missing:
        lines.append(f"  Thermal conductivity        — NOT READY (missing structural: {struct_missing})")
    else:
        lines.append("  Thermal conductivity        — NOT READY (no k data; run Stage 3 or override)")

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
    fiber_name: str = "",
    polymer_name: str = "",
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

    Two ways to specify the material system — pick ONE:
      1. card_id >= 0: load everything from a saved material card (fiber + polymer
         + microstructure + any inferred constituent properties). Best option when
         Stage 1 has already been run. Any override arguments still apply.
      2. fiber_name + polymer_name (no card): load fiber and polymer datasheet
         properties by name. You MUST then provide all microstructure fields
         explicitly via the override arguments (a11, a22, a12, a13, a23,
         fiber_massfrac, ar). Off-diagonal terms default to 0.0 if not provided.

    Any argument set to a value != -1.0 overrides what was loaded from the card or
    datasheet. Use this for what-if scenarios — e.g. change a11 while keeping
    everything else from a saved card.

    Thermoelastic prediction (CTE11, CTE22, CTE33) runs automatically when
    f_cte1_per_K, f_cte2_per_K, and m_cte_per_K are all available — either
    from a card with Stage 2 complete or from explicit override arguments.

    Units: moduli in MPa, CTE in 1/K (NOT ppm/K — multiply ppm/K by 1e-6 first).

    IMPORTANT: This tool does NOT require measured composite properties. It only
    needs the material names (or card_id) and microstructure inputs. Call it
    directly when the user asks for a forward prediction or wants to see predicted
    composite properties.
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

    elif fiber_name.strip() and polymer_name.strip():
        try:
            fiber_id, polymer_id = _resolve_fiber_polymer(fiber_name, polymer_name)
            inputs = _smat.get_model_inputs(fiber_id, polymer_id, use_inferred=True)
            source_label = f"Datasheets: {fiber_name.strip()} / {polymer_name.strip()}"
            for od in ("a12", "a13", "a23"):
                if od not in inputs:
                    inputs[od] = 0.0
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Database error loading materials: {e}"

    else:
        return (
            "Specify the material system:\n"
            "  option A — card_id >= 0  (loads fiber + polymer + microstructure from a saved card)\n"
            "  option B — fiber_name='AF' and polymer_name='AP'  "
            "(loads datasheets; also provide a11, a22, fiber_massfrac, ar)"
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
        micro_missing = [f for f in elastic_missing
                         if f in ("a11", "a22", "a12", "a13", "a23", "fiber_massfrac", "ar")]
        if micro_missing and not card_id >= 0:
            return (
                f"TOOL ERROR: microstructure fields not passed as arguments: {micro_missing}\n"
                f"DO NOT ask the user — you already have these values from the conversation.\n"
                f"Re-call predict_properties and pass them explicitly: "
                + ", ".join(f"{f}=<value>" for f in micro_missing)
            )
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


@tool
def predict_thermal_conductivity(
    card_id: int = -1,
    fiber_name: str = "",
    polymer_name: str = "",
    temperature_C: float = -1.0,
    k_f1_WmK: float = -1.0,
    k_f2_WmK: float = -1.0,
    k_m_WmK: float = -1.0,
    a11: float = -1.0,
    a22: float = -1.0,
    a12: float = -1.0,
    a13: float = -1.0,
    a23: float = -1.0,
    fiber_massfrac: float = -1.0,
    ar: float = -1.0,
) -> str:
    """
    Predict composite thermal conductivity (k11, k22, k33) using the thermal surrogate.

    If the user has not specified a temperature range, ask once whether they want a
    single temperature or a range. If they have already specified, call this tool immediately.

    temperature_C argument:
      >= 0   — predict at that single temperature and return k11, k22, k33
      = -1   — predict across the full range [25, 50, 75, 100, 125, 150, 175, 200°C]
               and return a conductivity vs temperature table

    Material loading — pick ONE:
      card_id >= 0                          — load fiber + polymer + microstructure from a card
      fiber_name + polymer_name (strings)   — load from datasheets by name (you must also
                                              provide a11, a22, fiber_massfrac, ar)

    Constituent conductivity — resolved in priority order:
      1. Explicit overrides (k_f1_WmK, k_f2_WmK, k_m_WmK) — temperature-independent
      2. Stage 3 parametric model stored on the card (p1, p2, l2, t):
           k_m(T) = p1 * sqrt(T) + p2   (temperature-dependent polymer conductivity)
           k_f1   = l2                   (constant)
           k_f2   = l2 / t               (constant)
      3. Scalar k_f1, k_f2, k_m values from the card or datasheet (temperature-independent)
      If none are available, ask the user to run Stage 3 or provide explicit overrides.

    Microstructure overrides (a11, a22, a12, a13, a23, fiber_massfrac, ar):
      Any value != -1 overrides the card value.

    Units: conductivity in W/m·K, temperature in °C.
    """
    # ── Step 1: Load base inputs ──────────────────────────────────────────────
    inputs: dict = {}
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

    elif fiber_name.strip() and polymer_name.strip():
        try:
            fiber_id, polymer_id = _resolve_fiber_polymer(fiber_name, polymer_name)
            inputs = _smat.get_model_inputs(fiber_id, polymer_id, use_inferred=True)
            source_label = f"Datasheets: {fiber_name.strip()} / {polymer_name.strip()}"
            for od in ("a12", "a13", "a23"):
                if od not in inputs:
                    inputs[od] = 0.0
        except ValueError as e:
            return str(e)
        except Exception as e:
            return f"Database error loading materials: {e}"

    else:
        return (
            "Specify the material system:\n"
            "  option A — card_id >= 0  (loads fiber + polymer + microstructure from a saved card)\n"
            "  option B — fiber_name='AF' and polymer_name='AP'  "
            "(also provide a11, a22, fiber_massfrac, ar)"
        )

    # ── Step 2: Ensure density and field-name aliases ─────────────────────────
    if "rho_f" not in inputs and "fiber_density" in inputs:
        inputs["rho_f"] = inputs["fiber_density"]
    if "fiber_density" not in inputs and "rho_f" in inputs:
        inputs["fiber_density"] = inputs["rho_f"]
    if "rho_m" not in inputs and "matrix_density" in inputs:
        inputs["rho_m"] = inputs["matrix_density"]
    if "matrix_density" not in inputs and "rho_m" in inputs:
        inputs["matrix_density"] = inputs["rho_m"]
    if "w_f" not in inputs and "fiber_massfrac" in inputs:
        inputs["w_f"] = inputs["fiber_massfrac"]
    if "ar_f" not in inputs and "ar" in inputs:
        inputs["ar_f"] = inputs["ar"]

    # ── Step 3: Apply structural overrides ────────────────────────────────────
    _struct_map = [
        ("a11", a11), ("a22", a22), ("a12", a12), ("a13", a13), ("a23", a23),
        ("fiber_massfrac", fiber_massfrac), ("ar", ar),
    ]
    for field, val in _struct_map:
        if val != -1.0:
            prev = inputs.get(field)
            inputs[field] = val
            if field == "fiber_massfrac":
                inputs["w_f"] = val
            elif field == "ar":
                inputs["ar_f"] = val
            if prev is not None and abs(float(prev) - val) > 1e-12:
                overrides_applied.append(f"  {field}: {float(prev):.4g} → {val:.4g}")
            else:
                overrides_applied.append(f"  {field} = {val:.4g}  [provided]")

    # ── Step 4: Collect explicit k overrides ─────────────────────────────────
    explicit_k: dict = {}
    if k_f1_WmK != -1.0:
        explicit_k["k_f1"] = k_f1_WmK
        overrides_applied.append(f"  k_f1 = {k_f1_WmK:.4g} W/m·K  [override]")
    if k_f2_WmK != -1.0:
        explicit_k["k_f2"] = k_f2_WmK
        overrides_applied.append(f"  k_f2 = {k_f2_WmK:.4g} W/m·K  [override]")
    if k_m_WmK != -1.0:
        explicit_k["k_m"] = k_m_WmK
        overrides_applied.append(f"  k_m = {k_m_WmK:.4g} W/m·K  [override]")
    use_explicit_k = len(explicit_k) == 3

    # ── Step 5: Check availability ────────────────────────────────────────────
    struct_missing = [f for f in _THERMAL_STRUCTURAL_REQUIRED if f not in inputs]
    has_parametric = all(inputs.get(p) is not None for p in ("p1", "p2", "k_f1", "k_f2"))
    has_scalar_k   = all(inputs.get(k) is not None for k in ("k_f1", "k_f2", "k_m"))
    k_available    = use_explicit_k or has_parametric or has_scalar_k

    # ── Step 6: Build header ──────────────────────────────────────────────────
    lines = [f"THERMAL CONDUCTIVITY PREDICTION\nSource: {source_label}"]

    if overrides_applied:
        lines.append("\nOverrides applied:")
        lines.extend(overrides_applied)

    if use_explicit_k:
        k_source = "explicit override values (temperature-independent)"
    elif has_parametric:
        p1, p2   = inputs["p1"], inputs["p2"]
        kf1, kf2 = inputs["k_f1"], inputs["k_f2"]
        k_source = (
            f"Stage 3 parametric model — "
            f"k_m(T) = {p1:.3e}·√T + {p2:.3e}  |  "
            f"k_f1 = {kf1:.4f},  k_f2 = {kf2:.4f} W/m·K (constant)"
        )
    elif has_scalar_k:
        k_source = "scalar k values from card/datasheet (temperature-independent)"
    else:
        k_source = "NOT AVAILABLE"

    lines.append(f"\nConstituent k source: {k_source}")

    if struct_missing:
        lines.append(f"\nCANNOT RUN: missing structural inputs: {struct_missing}")
        if card_id >= 0:
            lines.append("  Load a card with Stage 1 complete, or provide structural overrides.")
        else:
            lines.append("  Provide: a11, a22, a12, a13, a23, fiber_massfrac, ar")
        return "\n".join(lines)

    if not k_available:
        lines.append(
            "\nCANNOT RUN: no thermal conductivity data found for this material system.\n"
            "  Options:\n"
            "  - Run Stage 3 (thermal inverse) to infer p1, p2, l2, t from k vs T data\n"
            "  - Provide explicit overrides: k_f1_WmK, k_f2_WmK, k_m_WmK"
        )
        return "\n".join(lines)

    # ── Step 7: Determine temperature(s) ──────────────────────────────────────
    if temperature_C >= 0.0:
        temps = [temperature_C]
        mode  = "single"
    else:
        temps = _THERMAL_DEFAULT_TEMPS
        mode  = "matrix"

    # ── Step 8: Run predictions ───────────────────────────────────────────────
    try:
        results_by_T = []
        for T in temps:
            if use_explicit_k:
                kf1, kf2, km = explicit_k["k_f1"], explicit_k["k_f2"], explicit_k["k_m"]
            else:
                kf1, kf2, km = _resolve_k_at_T(inputs, T)

            thermal_inputs = {
                "k_f1": kf1, "k_f2": kf2, "k_m": km,
                "ar_f":  inputs["ar_f"],
                "w_f":   inputs["w_f"],
                "rho_f": inputs["rho_f"],
                "rho_m": inputs["rho_m"],
                "a11":   inputs["a11"],  "a22": inputs["a22"],
                "a12":   inputs["a12"],  "a13": inputs["a13"],  "a23": inputs["a23"],
            }
            pred = _sfwd.run_forward("thermal", thermal_inputs)
            results_by_T.append((T, kf1, kf2, km, pred))

    except Exception as e:
        return "\n".join(lines) + f"\n\nThermal model error: {e}"

    # ── Step 9: Format output ─────────────────────────────────────────────────
    if mode == "single":
        T, kf1, kf2, km, pred = results_by_T[0]
        lines.append(f"\nPREDICTION AT T = {T:.1f}°C")
        lines.append(
            f"  Constituent k:  k_f1 = {kf1:.4f},  k_f2 = {kf2:.4f},  "
            f"k_m = {km:.5f}  [W/m·K]"
        )
        lines.append("\n  Composite thermal conductivity:")
        for label, key in [
            ("k11 (print direction)", "k11"),
            ("k22 (transverse)",      "k22"),
            ("k33 (out-of-plane)",    "k33"),
        ]:
            v = pred.get(key)
            if v is not None:
                lines.append(f"    {label:<26} = {v:.5f} W/m·K")
    else:
        lines.append(
            f"\nTEMPERATURE MATRIX\n"
            f"  {'T (°C)':<8}  {'k_m (W/m·K)':<14}  "
            f"{'k11':<10}  {'k22':<10}  {'k33':<10}"
        )
        lines.append("  " + "-" * 58)
        for T, kf1, kf2, km, pred in results_by_T:
            k11 = pred.get("k11", float("nan"))
            k22 = pred.get("k22", float("nan"))
            k33 = pred.get("k33", float("nan"))
            lines.append(
                f"  {T:<8.1f}  {km:<14.5f}  {k11:<10.5f}  {k22:<10.5f}  {k33:<10.5f}"
            )
        if has_parametric:
            lines.append(
                f"\n  k_f1 = {results_by_T[0][1]:.4f} W/m·K (constant),  "
                f"k_f2 = {results_by_T[0][2]:.4f} W/m·K (constant)\n"
                f"  k_m varies with T: k_m(T) = {inputs['p1']:.3e}·√T + {inputs['p2']:.3e}"
            )

    return "\n".join(lines)


# ── Parameter sweep tool ─────────────────────────────────────────────────────

@tool
def sweep_parameter(
    card_id: int,
    parameter: str,
    values: list[float],
    target_property: str = "E1",
) -> str:
    """
    Sweep one microstructure or constituent parameter over a list of values
    and show how a target composite property changes. Use for what-if analysis,
    e.g. finding the aspect ratio that gives E1 = 15 GPa.

    Parameters
    ----------
    card_id         : card to base the sweep on
    parameter       : ar | a11 | a22 | fiber_massfrac | matrix_modulus |
                      matrix_poisson | f_cte1 | f_cte2 | m_cte
    values          : list of values to try, e.g. [10, 12, 14, 16, 18, 20]
    target_property : output to highlight, e.g. "E1", "E2", "G12", "CTE11"
    """
    try:
        result = _sfwd.sweep_parameter(card_id, parameter, values, target_property)
    except Exception as e:
        return f"SWEEP ERROR: {e}"

    prop = result["target_property"]
    rows = result["rows"]

    lines = [f"Sweep: {result['parameter']} → {prop}  (Card #{card_id})\n"]
    lines.append(f"{'Value':>10}  {prop:>14}")
    lines.append("-" * 28)
    for row in rows:
        val  = row["value"]
        pval = row["properties"].get(prop.lower()) or row["properties"].get(prop)
        pstr = f"{pval:,.2f}" if pval is not None else "N/A"
        lines.append(f"{val:>10.3f}  {pstr:>14}")

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


# ── Inverse solver tools ─────────────────────────────────────────────────────

import db.db as _db


def _resolve_fiber_polymer(fiber_name: str, polymer_name: str) -> tuple[int, int]:
    """Resolve fiber and polymer names to DB IDs. Raises ValueError if not found."""
    fibers   = {f["name"].lower(): f["id"] for f in _db.get_all_fibers()}
    polymers = {p["name"].lower(): p["id"] for p in _db.get_all_polymers()}
    fn = fiber_name.strip().lower()
    pn = polymer_name.strip().lower()
    if fn not in fibers:
        raise ValueError(f"Fiber '{fiber_name}' not found. Available: {', '.join(fibers)}")
    if pn not in polymers:
        raise ValueError(f"Polymer '{polymer_name}' not found. Available: {', '.join(polymers)}")
    return fibers[fn], polymers[pn]


def _resolve_material_ids(
    fiber_name: str, polymer_name: str, printer_name: str
) -> tuple[int, int, int]:
    """Resolve material names to DB IDs. Raises ValueError with a helpful message if not found."""
    fibers   = {f["name"].lower(): f["id"] for f in _db.get_all_fibers()}
    polymers = {p["name"].lower(): p["id"] for p in _db.get_all_polymers()}
    printers = {p["name"].lower(): p["id"] for p in _db.get_all_printers()}

    fn = fiber_name.strip().lower()
    pn = polymer_name.strip().lower()
    rn = printer_name.strip().lower()

    if fn not in fibers:
        raise ValueError(
            f"Fiber '{fiber_name}' not found. Available: {', '.join(f['name'] for f in _db.get_all_fibers())}"
        )
    if pn not in polymers:
        raise ValueError(
            f"Polymer '{polymer_name}' not found. Available: {', '.join(p['name'] for p in _db.get_all_polymers())}"
        )
    if rn not in printers:
        raise ValueError(
            f"Printer '{printer_name}' not found. Available: {', '.join(p['name'] for p in _db.get_all_printers())}"
        )

    return fibers[fn], polymers[pn], printers[rn]


@tool
def run_elastic_inverse(
    fiber_name: str,
    polymer_name: str,
    printer_name: str,
    # Composite elastic measurements (omit or pass -1.0 to exclude)
    E1_MPa: float = -1.0,
    E2_MPa: float = -1.0,
    E3_MPa: float = -1.0,
    G12_MPa: float = -1.0,
    G13_MPa: float = -1.0,
    G23_MPa: float = -1.0,
    nu12: float = -1.0,
    nu13: float = -1.0,
    nu23: float = -1.0,
    # Measurement uncertainties (1-sigma, same units; use 0.0 if unknown)
    E1_sigma_MPa: float = 0.0,
    E2_sigma_MPa: float = 0.0,
    E3_sigma_MPa: float = 0.0,
    G12_sigma_MPa: float = 0.0,
    G13_sigma_MPa: float = 0.0,
    G23_sigma_MPa: float = 0.0,
    nu12_sigma: float = 0.0,
    nu13_sigma: float = 0.0,
    nu23_sigma: float = 0.0,
    # Known microstructure (pass values to fix them; omit/None to let solver infer)
    ar: Optional[float] = None,
    fiber_massfrac: Optional[float] = None,
    a11: Optional[float] = None,
    a22: Optional[float] = None,
    a12: float = 0.0,
    a13: float = 0.0,
    a23: float = 0.0,
    card_name: str = "",
) -> str:
    """
    Run Stage 1 elastic inverse: infer microstructure (a11, a22, etc.) and
    in-situ constituent properties (matrix_modulus) from measured composite
    elastic properties.

    Pass material NAMES (not IDs) — e.g. fiber_name="AF", polymer_name="AP",
    printer_name="CAMRI". IDs are resolved automatically.

    If the user already knows some microstructure values (from CT, datasheet, etc.),
    pass them as ar/fiber_massfrac/a11/a22/a12/a13/a23 — they will be fixed and not
    inferred. Omit (or pass None) to let the solver infer them.

    All elastic measurement arguments (E1_MPa, etc.) must be in MPa.
    Sigma arguments are 1-sigma uncertainty in the same units. Use 0.0 if unknown.

    Results are held in memory — call save_to_card() if the user wants to persist.
    Does NOT save automatically.
    """
    # ── Resolve names to IDs ──────────────────────────────────────────────────
    try:
        fiber_id, polymer_id, printer_id = _resolve_material_ids(
            fiber_name, polymer_name, printer_name
        )
    except ValueError as e:
        return f"Material lookup error: {e}"

    # ── Build target_outputs and sigmas ───────────────────────────────────────
    _meas_map = [
        ("E1",   E1_MPa,  E1_sigma_MPa),
        ("E2",   E2_MPa,  E2_sigma_MPa),
        ("E3",   E3_MPa,  E3_sigma_MPa),
        ("G12",  G12_MPa, G12_sigma_MPa),
        ("G13",  G13_MPa, G13_sigma_MPa),
        ("G23",  G23_MPa, G23_sigma_MPa),
        ("nu12", nu12,    nu12_sigma),
        ("nu13", nu13,    nu13_sigma),
        ("nu23", nu23,    nu23_sigma),
    ]
    targets: dict = {}
    sigmas: dict  = {}
    for name, val, sig in _meas_map:
        if val != -1.0:
            targets[name] = val
            if sig > 0.0:
                sigmas[name] = sig

    if not targets:
        return (
            "No measurements provided. Supply at least E1_MPa and one of E2_MPa or E3_MPa.\n"
            "Example: run_elastic_inverse(fiber_id=1, polymer_id=1, printer_id=1,\n"
            "         E1_MPa=45000, E2_MPa=12000, E3_MPa=10000)"
        )

    # ── Load datasheet inputs ─────────────────────────────────────────────────
    try:
        datasheet = _smat.get_model_inputs(fiber_id, polymer_id)
    except Exception as e:
        return f"Error loading material datasheets (fiber_id={fiber_id}, polymer_id={polymer_id}): {e}"

    # ── Collect user-provided (fixed) microstructure ──────────────────────────
    _micro_args = {
        "ar": ar, "fiber_massfrac": fiber_massfrac,
        "a11": a11, "a22": a22, "a12": a12, "a13": a13, "a23": a23,
    }
    known_micro = {k: v for k, v in _micro_args.items() if v is not None}

    # ar and fiber_massfrac default to fixed at the datasheet value.
    # Only become free if the user explicitly requests inference (agent omits them → None).
    for field in ("ar", "fiber_massfrac"):
        if field not in known_micro and field in datasheet:
            known_micro[field] = datasheet[field]

    # Determine free variables first, then build fixed_inputs as everything else.
    # matrix_poisson is only identifiable when shear/Poisson measurements are present;
    # without them it stays fixed at the datasheet value.
    has_shear = bool(set(targets.keys()) & _SHEAR_POISSON_MEASUREMENTS)
    base_free = _STAGE1_FREE if has_shear else _STAGE1_FREE_WITHOUT_POISSON

    # Free variables = chosen base set minus whatever the user already fixed
    free_vars = [v for v in base_free if v not in known_micro]

    # Fixed inputs = full datasheet minus the actual free vars + user-provided microstructure.
    # This ensures fields like matrix_poisson are always present in fixed_inputs when not free.
    free_set = set(free_vars)
    fixed_inputs = {k: v for k, v in datasheet.items() if k not in free_set}
    fixed_inputs.update(known_micro)

    # ── Run solver ────────────────────────────────────────────────────────────
    # Strip keys the model doesn't know (e.g. rho_f, rho_m alias fields from DB)
    _model_fields = set(_sfwd.get_input_fields("elastic"))
    fixed_inputs = {k: v for k, v in fixed_inputs.items() if k in _model_fields}

    bounds = {k: _DEFAULT_BOUNDS[k] for k in free_vars if k in _DEFAULT_BOUNDS}

    # Initial guess: orientation defaults from problem.json; material-specific fields
    # (matrix_modulus, matrix_poisson) fall through to the datasheet so they match
    # the GUI which populates rows from the DB.
    _PROBLEM_JSON_INIT = {
        "a11": 0.6, "a22": 0.1, "a12": 0.0, "a13": 0.0, "a23": 0.0,
        "fiber_massfrac": 0.20, "ar": 20.0,
    }
    init_vals = []
    for k in free_vars:
        if k in _PROBLEM_JSON_INIT:
            init_vals.append(_PROBLEM_JSON_INIT[k])
        elif k in datasheet:
            init_vals.append(float(datasheet[k]))
        elif k in fixed_inputs:
            init_vals.append(float(fixed_inputs[k]))
        elif k in _DEFAULT_BOUNDS:
            lo, hi = _DEFAULT_BOUNDS[k]
            init_vals.append((lo + hi) / 2.0)
        else:
            init_vals.append(0.0)

    # Enable epsilon-insensitive loss when measurement sigmas are provided.
    # This creates a dead-zone of ±sigma around each target: residuals inside
    # it contribute zero loss. Measurements with larger sigma exert less pull
    # on the solution — matching the physical meaning of measurement uncertainty.
    solver_cfg = dict(_ELASTIC_SOLVER_CFG)
    if sigmas:
        solver_cfg["use_epsilon_loss"] = True

    try:
        inv = _sinv.run_inverse(
            model_name="elastic",
            fixed_inputs=fixed_inputs,
            free_inputs=free_vars,
            bounds=bounds,
            target_outputs=targets,
            sigmas=sigmas if sigmas else None,
            solver_cfg=solver_cfg,
            init_vals=init_vals,
        )
    except Exception as e:
        return f"Solver error: {e}"

    # ── Cache result for save_to_card ─────────────────────────────────────────
    # known_micro is already in fixed_inputs (via fixed_inputs.update(known_micro)),
    # so save_inverse_result sees it there with "inputted" provenance. Do NOT merge
    # it into opt_free — that would cause user-provided mf/ar to be tagged "inferred".
    opt_free = inv["opt_free"]
    _pending_save.clear()
    _pending_save["result"] = {
        "model":             "elastic",
        "opt_free":          inv["opt_free"],
        "fixed_inputs":      fixed_inputs,
        "predicted_outputs": inv["predicted_outputs"],
        "target_outputs":    inv["target_outputs"],
        "sigmas":            sigmas,
        "final_error":       inv["final_error"],
        "solver_cfg":        inv["solver_cfg"],
    }
    _pending_save["meta"] = {
        "fiber_id":   fiber_id,
        "polymer_id": polymer_id,
        "printer_id": printer_id,
        "card_name":  card_name,
    }

    # ── Format result string ──────────────────────────────────────────────────
    free = opt_free
    pred = inv["predicted_outputs"]
    err  = inv["final_error"]

    using_eps_loss = bool(sigmas)

    def _fmt_micro(key: str, label: str, fmt: str = ".4f", suffix: str = "") -> str:
        if key in known_micro:
            return f"  {label} = {known_micro[key]:{fmt}}{suffix}  (fixed — user provided)"
        val = free.get(key, float("nan"))
        return f"  {label} = {val:{fmt}}{suffix}  (inferred)"

    lines = [
        "ELASTIC INVERSE — COMPLETE",
        "",
        "Microstructure:",
        _fmt_micro("a11", "a11           ", suffix="  (alignment — print direction)"),
        _fmt_micro("a22", "a22           ", suffix="  (transverse)"),
        _fmt_micro("a12", "a12           "),
        _fmt_micro("a13", "a13           "),
        _fmt_micro("a23", "a23           "),
        _fmt_micro("fiber_massfrac", "fiber_massfrac"),
        _fmt_micro("ar",  "aspect_ratio  ", fmt=".2f"),
        "",
        "Inferred constituent properties:",
        f"  matrix_modulus  = {free.get('matrix_modulus', float('nan')):.1f} MPa  (inferred)",
        f"  matrix_poisson  = {fixed_inputs.get('matrix_poisson', free.get('matrix_poisson', float('nan'))):.4f}"
        + ("  (inferred — shear measurements present)" if has_shear else "  (fixed — datasheet; add G12/nu12 to infer)"),
        "",
    ]

    if using_eps_loss:
        lines.append("Model fit vs targets  (measurement uncertainty active — predictions within ±σ are acceptable):")
        all_within = True
        for meas_name, t_val in targets.items():
            p_val   = pred.get(meas_name, float("nan"))
            sig     = sigmas.get(meas_name, 0.0)
            resid   = abs(p_val - t_val)
            unit    = " MPa" if meas_name.startswith(("E", "G")) else ""
            within  = resid <= sig * 1.001  # 0.1% tolerance for float edge cases
            if not within:
                all_within = False
            status  = "✓ within σ" if within else f"✗ outside σ by {resid - sig:.1f}{unit}"
            lines.append(
                f"  {meas_name:<5}  predicted={p_val:.1f}{unit}  target={t_val:.1f}{unit}"
                f"  residual={resid:.1f}{unit}  σ={sig:.1f}{unit}  {status}"
            )
        verdict = "All measurements satisfied within uncertainty." if all_within \
                  else "WARNING: some predictions fall outside measurement uncertainty — check inputs."
        lines += ["", verdict]
    else:
        # Standard MSE — report normalized fit error
        err_label = (
            "good fit" if err < 0.01 else
            "acceptable" if err < 0.05 else
            "POOR — check measurements or material assignment"
        )
        lines.append(f"fit_error = {err:.5f}  ({err_label})")
        lines.append("Model fit vs targets:")
        for meas_name, t_val in targets.items():
            p_val = pred.get(meas_name, float("nan"))
            unit  = " MPa" if meas_name.startswith(("E", "G")) else ""
            diff  = abs(p_val - t_val) / max(abs(t_val), 1e-9) * 100
            lines.append(
                f"  {meas_name:<5}  predicted={p_val:.1f}{unit}  target={t_val:.1f}{unit}  diff={diff:.2f}%"
            )

    lines += ["", "Results are in memory. Call save_to_card() to persist, or discard."]
    return "\n".join(lines)


@tool
def run_thermoelastic_inverse(
    card_id: int,
    CTE11_per_K: float,
    CTE22_per_K: float,
    CTE33_per_K: float = -1.0,
    CTE11_sigma_per_K: float = 0.0,
    CTE22_sigma_per_K: float = 0.0,
    CTE33_sigma_per_K: float = 0.0,
) -> str:
    """
    Run Stage 2 thermoelastic inverse: infer fiber CTEs (f_cte1, f_cte2) and
    matrix CTE (m_cte) from measured composite thermal expansion coefficients.

    Requires Stage 1 (elastic inverse) to be saved on this card first.
    Loads microstructure and matrix modulus from the card automatically —
    those values are held fixed; do not re-enter them.

    card_id: the card_id returned by save_to_card() after Stage 1.
    CTE11_per_K: measured composite CTE along the print direction (1/K). Required.
    CTE22_per_K: measured composite CTE transverse to print direction (1/K). Required.
    CTE33_per_K: measured composite CTE out-of-plane (1/K). Pass -1.0 to exclude.
    CTE11/22/33_sigma_per_K: 1-sigma measurement uncertainty (1/K). Use 0.0 if unknown.
      Convert ppm/K → 1/K before passing: ppm/K × 1e-6.

    Inferred CTEs are constituent (printer-independent) and will be saved
    globally — reusable for any card with the same fiber and polymer.

    Results held in memory. Call save_to_card(card_id=<same id>) to persist.
    """
    # ── Load card metadata ────────────────────────────────────────────────────
    try:
        card = _scards.load_card(card_id)
    except Exception as e:
        return f"Card lookup error: {e}"

    if card is None:
        return f"Card id={card_id} not found. Use list_cards() to see available cards."

    cfg        = card["config"]
    fiber_id   = cfg["fiber_id"]
    polymer_id = cfg["polymer_id"]
    printer_id = cfg.get("printer_id")

    # ── Load Stage 1 resolved inputs ──────────────────────────────────────────
    try:
        all_inputs = _scards.load_card_inputs(card_id)
    except Exception as e:
        return f"Error loading card inputs: {e}"

    if "matrix_modulus" not in all_inputs:
        return (
            f"Stage 1 (elastic inverse) has not been saved for card id={card_id}. "
            "Run run_elastic_inverse first, call save_to_card, then run Stage 2."
        )
    if "a11" not in all_inputs:
        return (
            f"No microstructure snapshot found for card id={card_id}. "
            "Run run_elastic_inverse first, save to this card, then run Stage 2."
        )

    # ── Targets and sigmas ────────────────────────────────────────────────────
    targets: dict = {"CTE11": CTE11_per_K, "CTE22": CTE22_per_K}
    if CTE33_per_K != -1.0:
        targets["CTE33"] = CTE33_per_K
    sigmas: dict = {}
    if CTE11_sigma_per_K > 0.0:
        sigmas["CTE11"] = CTE11_sigma_per_K
    if CTE22_sigma_per_K > 0.0:
        sigmas["CTE22"] = CTE22_sigma_per_K
    if CTE33_per_K != -1.0 and CTE33_sigma_per_K > 0.0:
        sigmas["CTE33"] = CTE33_sigma_per_K

    # ── Fixed inputs (Stage 1 values, filtered to thermoelastic model fields) ─
    free_set   = set(_TE_FREE)
    _te_fields = set(_sfwd.get_input_fields("thermoelastic"))
    fixed_inputs = {
        k: v for k, v in all_inputs.items()
        if k not in free_set and k in _te_fields
    }

    # ── Bounds and initial values ─────────────────────────────────────────────
    bounds    = {k: _DEFAULT_BOUNDS[k] for k in _TE_FREE if k in _DEFAULT_BOUNDS}
    init_vals = [_TE_INIT.get(k, 0.0) for k in _TE_FREE]

    # ── Solver config ─────────────────────────────────────────────────────────
    solver_cfg = dict(_ELASTIC_SOLVER_CFG)
    if sigmas:
        solver_cfg["use_epsilon_loss"] = True

    # ── Run inverse ───────────────────────────────────────────────────────────
    try:
        inv = _sinv.run_inverse(
            model_name="thermoelastic",
            fixed_inputs=fixed_inputs,
            free_inputs=_TE_FREE,
            bounds=bounds,
            target_outputs=targets,
            sigmas=sigmas if sigmas else None,
            solver_cfg=solver_cfg,
            init_vals=init_vals,
        )
    except Exception as e:
        return f"Solver error: {e}"

    # ── Cache for save_to_card ────────────────────────────────────────────────
    _pending_save.clear()
    _pending_save["result"] = {
        "model":             "thermoelastic",
        "opt_free":          inv["opt_free"],
        "fixed_inputs":      fixed_inputs,
        "predicted_outputs": inv["predicted_outputs"],
        "target_outputs":    inv["target_outputs"],
        "sigmas":            sigmas,
        "final_error":       inv["final_error"],
        "solver_cfg":        inv["solver_cfg"],
    }
    _pending_save["meta"] = {
        "fiber_id":   fiber_id,
        "polymer_id": polymer_id,
        "printer_id": printer_id,
        "card_name":  cfg.get("name", ""),
    }

    # ── Format result ─────────────────────────────────────────────────────────
    free = inv["opt_free"]
    pred = inv["predicted_outputs"]
    err  = inv["final_error"]

    def _fmt_cte(val: float) -> str:
        return f"{val * 1e6:.4f} ppm/K  ({val:.4e} 1/K)"

    err_label = (
        "good fit"   if err < 0.01 else
        "acceptable" if err < 0.05 else
        "POOR — check measurements or material assignment"
    )

    lines = [
        "THERMOELASTIC INVERSE — COMPLETE",
        "",
        "Inferred constituent CTEs:",
        f"  f_cte1 (fiber axial CTE)       = {_fmt_cte(free['f_cte1'])}",
        f"  f_cte2 (fiber transverse CTE)  = {_fmt_cte(free['f_cte2'])}",
        f"  m_cte  (matrix CTE)            = {_fmt_cte(free['m_cte'])}",
        "",
        f"fit_error = {err:.5f}  ({err_label})",
        "Model fit vs targets:",
    ]

    for meas_name, t_val in targets.items():
        p_val = pred.get(meas_name, float("nan"))
        sig   = sigmas.get(meas_name, 0.0)
        diff  = abs(p_val - t_val) / max(abs(t_val), 1e-9) * 100
        sig_str = f"  σ={sig * 1e6:.4f} ppm/K" if sig > 0.0 else ""
        lines.append(
            f"  {meas_name:<6}  predicted={p_val * 1e6:.4f} ppm/K"
            f"  target={t_val * 1e6:.4f} ppm/K  diff={diff:.2f}%{sig_str}"
        )

    lines += [
        "",
        "These CTEs are printer-independent and will be saved globally",
        "(reusable for any card with the same fiber and polymer).",
        "",
        f"Results in memory. Call save_to_card(card_id={card_id}) to persist.",
    ]
    return "\n".join(lines)


# ── CSV helper for run_thermal_inverse ───────────────────────────────────────

def _parse_thermal_csv(csv_path: str):
    """Parse a K vs T CSV file. Returns (temperatures, K_data).

    temperatures : np.ndarray, shape (N,), °C
    K_data       : {"K11": array|None, "K22": array|None, "K33": array|None}

    Accepted column names (case-insensitive):
      Temperature : temperature_c  temperature  t  temp_c  temp
      K11         : k11_wmk  k11
      K22         : k22_wmk  k22
      K33         : k33_wmk  k33
    """
    import csv as _csv
    import numpy as np

    _T_ALIASES = {"temperature_c", "temperature", "t", "temp_c", "temp"}
    _K_ALIASES = {
        "K11": {"k11_wmk", "k11"},
        "K22": {"k22_wmk", "k22"},
        "K33": {"k33_wmk", "k33"},
    }

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(path, newline="") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        raw = {h.strip().lower(): h for h in reader.fieldnames}

        t_col = next((raw[a] for a in _T_ALIASES if a in raw), None)
        if t_col is None:
            raise ValueError(
                f"No temperature column found in CSV. Expected one of: {sorted(_T_ALIASES)}"
            )

        k_cols = {
            key: next((raw[a] for a in aliases if a in raw), None)
            for key, aliases in _K_ALIASES.items()
        }

        rows = list(reader)

    if not rows:
        raise ValueError("CSV is empty (no data rows).")

    temps = np.array([float(r[t_col]) for r in rows])
    K_data = {
        key: np.array([float(r[col]) for r in rows]) if col else None
        for key, col in k_cols.items()
    }

    if K_data["K11"] is None:
        raise ValueError("K11 column is required but was not found in the CSV.")

    return temps, K_data


@tool
def run_thermal_inverse(
    card_id: int,
    csv_path: str,
    n_restarts: int = 20,
) -> str:
    """
    Run Stage 3 thermal inverse: infer fiber conductivities (k_f1 = l2, k_f2 = l2/t)
    and polymer conductivity model parameters (p1, p2) where K_m(T) = p1·√T + p2.

    Requires Stage 1 (elastic inverse) saved on this card.
    Microstructure and material inputs are loaded from the card automatically.

    card_id    : card_id from Stage 1 save.
    csv_path   : absolute path to CSV file. Required columns:
                   temperature_C  (°C) and K11_WmK  (W/m·K)
                 Optional columns: K22_WmK, K33_WmK — improve fit if available.
                 Header row required; column names are case-insensitive.
    n_restarts : optimisation restarts (default 20).

    Results are held in memory. Call save_to_card(card_id=<same id>) to persist.
    """
    global _THERMAL_FWD_MODEL

    # ── Load card ─────────────────────────────────────────────────────────────
    try:
        card = _scards.load_card(card_id)
    except Exception as e:
        return f"Card lookup error: {e}"
    if card is None:
        return f"Card id={card_id} not found. Use list_cards() to see available cards."

    cfg        = card["config"]
    fiber_id   = cfg["fiber_id"]
    polymer_id = cfg["polymer_id"]
    printer_id = cfg.get("printer_id")

    # ── Load Stage 1 inputs ───────────────────────────────────────────────────
    try:
        all_inputs = _scards.load_card_inputs(card_id)
    except Exception as e:
        return f"Error loading card inputs: {e}"

    if "matrix_modulus" not in all_inputs or "a11" not in all_inputs:
        return (
            f"Stage 1 (elastic inverse) has not been saved for card id={card_id}. "
            "Run run_elastic_inverse first, save to this card, then run Stage 3."
        )

    # ── Build fixed_inputs (remap card field names to inverse_thermal.py names) ─
    fixed_inputs = {
        "ar_f":  all_inputs.get("ar",            all_inputs.get("ar_f")),
        "w_f":   all_inputs.get("fiber_massfrac", all_inputs.get("w_f")),
        "rho_f": all_inputs.get("rho_f"),
        "rho_m": all_inputs.get("rho_m"),
        "a11":   all_inputs["a11"],
        "a22":   all_inputs["a22"],
        "a12":   all_inputs.get("a12", 0.0),
        "a13":   all_inputs.get("a13", 0.0),
        "a23":   all_inputs.get("a23", 0.0),
    }
    missing = [k for k, v in fixed_inputs.items() if v is None]
    if missing:
        return f"Missing required fields from card: {missing}. Check that the card has complete Stage 1 data."

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    try:
        temperatures, K_data = _parse_thermal_csv(csv_path)
    except Exception as e:
        return f"CSV error: {e}"

    n_temps   = len(temperatures)
    k_present = [k for k, v in K_data.items() if v is not None]

    # ── Load thermal forward model ────────────────────────────────────────────
    try:
        if _THERMAL_FWD_MODEL is None:
            _THERMAL_FWD_MODEL = _sfwd.load_forward("thermal")
        predictor = _ithermal.make_batched_predictor(_THERMAL_FWD_MODEL)
    except Exception as e:
        return f"Error loading thermal model: {e}"

    # ── Run inverse estimation ────────────────────────────────────────────────
    try:
        best_params, best_loss = _ithermal.run_inverse_estimation(
            temperatures=temperatures,
            K_data=K_data,
            predictor=predictor,
            fixed_inputs=fixed_inputs,
            n_restarts=n_restarts,
        )
    except Exception as e:
        return f"Solver error: {e}"

    # ── Cache for save_to_card ────────────────────────────────────────────────
    _pending_save.clear()
    _pending_save["result"] = {
        "model":       "thermal",
        "best_params": best_params,
        "best_loss":   best_loss,
    }
    _pending_save["meta"] = {
        "fiber_id":   fiber_id,
        "polymer_id": polymer_id,
        "printer_id": printer_id,
        "card_name":  cfg.get("name", ""),
    }

    # ── Format output ─────────────────────────────────────────────────────────
    p1, p2, l2, t = best_params.p1, best_params.p2, best_params.l2, best_params.t
    k_f1 = l2
    k_f2 = l2 / t
    k_m_25 = p1 * (25.0 ** 0.5) + p2

    err_label = (
        "good fit"   if best_loss < 1e-4 else
        "acceptable" if best_loss < 1e-2 else
        "POOR — check measurements or material assignment"
    )

    lines = [
        "THERMAL INVERSE — COMPLETE",
        "",
        "Inferred constituent thermal conductivities:",
        f"  k_f1  (fiber longitudinal)  = {k_f1:.4f} W/m·K  (= l2)",
        f"  k_f2  (fiber transverse)    = {k_f2:.4f} W/m·K  (= l2/t)",
        f"  p1    (polymer scaling)      = {p1:.4e} W/m·K",
        f"  p2    (polymer offset)       = {p2:.4f} W/m·K",
        f"  k_m @ 25°C                  = {k_m_25:.4f} W/m·K",
        f"  t     (anisotropy ratio)     = {t:.4f}",
        "",
        f"fit_error = {best_loss:.2e}  ({err_label})",
        f"Data: {n_temps} temperature points, channels fitted: {', '.join(k_present)}",
        "",
        "These properties are printer-independent and will be saved globally",
        "(reusable for any card with the same fiber and polymer).",
        "",
        f"Results in memory. Call save_to_card(card_id={card_id}) to persist.",
    ]
    return "\n".join(lines)


@tool
def save_to_card(card_name: str = "", card_id: int = -1) -> str:
    """
    Save the most recent solver result to the database.

    card_name: name for the new card (e.g. "AF/AP CAMRI Stage1"). ALWAYS ask
               the user for a card name before calling this tool if card_id=-1.
    card_id = -1  → create a new material card using card_name
    card_id >= 0  → save to an existing card (updates it in place, card_name ignored)

    Only call this after:
      1. A solver tool (run_elastic_inverse, etc.) returned a successful result, AND
      2. The user has explicitly confirmed they want to save, AND
      3. You have asked for and received a card_name (when card_id=-1).

    Do NOT call automatically — always ask first.
    Returns the card_id so subsequent tools can reference it.
    """
    if not _pending_save:
        return (
            "No result in memory to save. "
            "Run run_elastic_inverse (or another inverse tool) first."
        )

    result = _pending_save["result"]
    meta   = _pending_save["meta"]

    # card_name: prefer the one passed here; fall back to what the solver captured
    name = card_name.strip() or meta.get("card_name", "")

    try:
        if result.get("model") == "thermal":
            if card_id == -1:
                return (
                    "Thermal results must be saved to an existing card. "
                    "Provide card_id (the same card_id used in run_thermal_inverse)."
                )
            _scards.save_thermal_result(
                result=result,
                fiber_id=meta["fiber_id"],
                polymer_id=meta["polymer_id"],
                card_id=card_id,
            )
            saved_id = card_id
        else:
            saved_id = _scards.save_inverse_result(
                result=result,
                fiber_id=meta["fiber_id"],
                polymer_id=meta["polymer_id"],
                printer_id=meta.get("printer_id"),
                card_id=None if card_id == -1 else card_id,
                card_name=name,
            )
    except Exception as e:
        return f"Save error: {e}"

    stage = result.get("model", "unknown")
    _pending_save.clear()

    return (
        f"Saved successfully.\n"
        f"  card_id   = {saved_id}\n"
        f"  card_name = {name}\n"
        f"  stage     = {stage}\n"
        f"Use card_id={saved_id} in get_card_status, predict_properties, "
        f"or subsequent inverse stages."
    )


@tool
def save_processing_conditions(
    card_id: int,
    bead_width: float = -1.0,
    bead_height: float = -1.0,
    nozzle_diameter: float = -1.0,
    print_speed: float = -1.0,
    notes: str = "",
) -> str:
    """
    Save printing process conditions to an existing material card.

    card_id:         the card to attach conditions to (from save_to_card)
    bead_width:      bead width in mm (-1 if unknown)
    bead_height:     bead height / layer thickness in mm (-1 if unknown)
    nozzle_diameter: nozzle diameter in mm (-1 if unknown)
    print_speed:     print speed in mm/s (-1 if unknown)
    notes:           any other process notes (temperature, infill pattern, etc.)

    Call this after save_to_card. At least one of the numeric fields or notes
    must be provided. Pass -1 for any value the user does not have.
    """
    if card_id < 0:
        return "card_id is required. Call save_to_card first to get a card_id."

    has_any = (bead_width >= 0 or bead_height >= 0 or
               nozzle_diameter >= 0 or print_speed >= 0 or bool(notes))
    if not has_any:
        return "No conditions provided. Pass at least one value or a notes string."

    try:
        _scards.save_processing_conditions(
            card_id=card_id,
            bead_width=bead_width      if bead_width      >= 0 else None,
            bead_height=bead_height    if bead_height     >= 0 else None,
            nozzle_diameter=nozzle_diameter if nozzle_diameter >= 0 else None,
            print_speed=print_speed    if print_speed     >= 0 else None,
            notes=notes,
        )
    except Exception as e:
        return f"Error saving processing conditions: {e}"

    lines = ["Processing conditions saved."]
    if bead_width      >= 0: lines.append(f"  bead_width      = {bead_width} mm")
    if bead_height     >= 0: lines.append(f"  bead_height     = {bead_height} mm")
    if nozzle_diameter >= 0: lines.append(f"  nozzle_diameter = {nozzle_diameter} mm")
    if print_speed     >= 0: lines.append(f"  print_speed     = {print_speed} mm/s")
    if notes:                lines.append(f"  notes           = {notes}")
    return "\n".join(lines)


# ── All tools passed to the LLM and ToolNode ─────────────────────────────────

TOOLS = [
    search_knowledge_base,
    list_materials,
    get_material_details,
    convert_fraction,
    list_cards,
    get_card_status,
    inspect_card_inputs,
    get_model_inputs_outputs,
    predict_properties,
    predict_thermal_conductivity,
    add_fiber,
    add_polymer,
    check_identifiability,
    sweep_parameter,
    run_elastic_inverse,
    run_thermoelastic_inverse,
    run_thermal_inverse,
    save_to_card,
    save_processing_conditions,
]
