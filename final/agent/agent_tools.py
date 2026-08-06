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
    predict_properties            — forward prediction (elastic + thermoelastic); accepts card, DB name, or raw constituent values
    predict_thermal_conductivity  — forward prediction (thermal) at one T or across a T range; accepts card, DB name, or raw constituent values
    compare_thermal_fit           — compare Stage 3 model predictions against experimental CSV data; reports per-channel predicted vs measured table with % errors
    add_fiber                     — add a new fiber to the material library
    add_polymer                   — add a new polymer to the material library
    update_fiber                  — edit fields of an existing fiber (nu23, moduli, CTE, k, etc.)
    update_polymer                — edit fields of an existing polymer (E, nu, CTE, k, etc.)
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
import core.services.service_thermal as _sthermal
import core.services.service_transfer as _stransfer


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

# Holds pipeline results across all 3 stages — populated by run_full_pipeline,
# consumed by save_to_card when this dict is non-empty.
# Structure: {"stage1": ..., "stage2": ..., "stage3": ..., "meta": {fiber_id, ...}}
_pending_pipeline: dict = {}

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
    # Intentionally wider than config/problem.json — the agent handles a broader
    # range of materials and printers than the GUI's single example problem.
    # a11: GUI uses [0.2, 0.81]; agent uses [0.50, 0.85] (excludes near-random)
    # a22: GUI uses [0.01, 0.2]; agent uses [0.01, 0.4] (allows more transverse)
    "a11":            (0.50,    0.85),
    "a22":            (0.01,    0.4),
    "a12":            (-0.10,   0.10),
    "a13":            (-0.10,   0.10),
    "a23":            (-0.10,   0.10),
    "fiber_massfrac": (0.05,    0.60),
    "ar":             (5.0,    100.0),
    "matrix_modulus": (2000.0, 5000.0),
    "matrix_poisson": (0.33,    0.42),
    "f_cte1":         (-4e-6,   4e-6),
    "f_cte2":         (7e-6,   15e-6),
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
    "f_cte2": 11.0e-6,   # midpoint of bounds [7, 15] ppm/K
    "m_cte":  75.0e-6,   # midpoint of bounds; polymers typically 50-100 ppm/K
}

# Stage 3 — thermal inverse forward model (lazy-loaded on first call)

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

    lines = [f"MATERIAL CARDS — {len(configs)} total (report all {len(configs)} to the user)\n"]
    for cfg in configs:
        fiber   = fibers.get(cfg["fiber_id"],   {})
        polymer = polymers.get(cfg["polymer_id"], {})

        printer_name = "no printer"
        if cfg.get("printer_id"):
            pr = printers.get(cfg["printer_id"], {})
            printer_name = pr.get("name", f"printer_id={cfg['printer_id']}")

        stages = _smat.get_completed_stages(cfg["id"], cfg["fiber_id"], cfg["polymer_id"])
        stage_str      = ", ".join(stages) if stages else "none"
        transferable   = {"elastic", "thermoelastic", "thermal"}.issubset(stages)
        transfer_str   = "yes" if transferable else f"no (missing: {', '.join(sorted({'elastic','thermoelastic','thermal'} - set(stages)))})"

        fiber_name   = fiber.get("name",   f"fiber_id={cfg['fiber_id']}")
        polymer_name = polymer.get("name", f"polymer_id={cfg['polymer_id']}")

        lines.append(
            f"  card_id={cfg['id']}  \"{cfg['name']}\"\n"
            f"    Fiber:   {fiber_name}\n"
            f"    Polymer: {polymer_name}\n"
            f"    Printer: {printer_name}\n"
            f"    Completed stages: {stage_str}\n"
            f"    Transferable: {transfer_str}\n"
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
    transferable = {"elastic", "thermoelastic", "thermal"}.issubset(stages)
    transfer_str = "yes" if transferable else f"no (missing: {', '.join(sorted({'elastic','thermoelastic','thermal'} - set(stages)))})"

    lines = [
        f"CARD STATUS — \"{cfg['name']}\"  (card_id={card_id})",
        f"  Fiber:   {fiber_name}",
        f"  Polymer: {polymer_name}",
        f"  Printer: {printer_name}",
        f"  Completed stages: {', '.join(stages) if stages else 'none — no inverse runs saved yet'}",
        f"  Transferable: {transfer_str}",
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
    fiber_E1_MPa: float = -1.0,
    fiber_E2_MPa: float = -1.0,
    fiber_G12_MPa: float = -1.0,
    fiber_nu12: float = -1.0,
    fiber_nu23: float = -1.0,
    fiber_density_kg_m3: float = -1.0,
    matrix_density_kg_m3: float = -1.0,
) -> str:
    """
    Run elastic (and optionally thermoelastic) forward prediction.

    Three ways to specify the material system — pick ONE:
      1. card_id >= 0: load everything from a saved material card (fiber + polymer
         + microstructure + any inferred constituent properties). Best option when
         Stage 1 has already been run. Any override arguments still apply.
      2. fiber_name + polymer_name (no card): load fiber and polymer datasheet
         properties by name from the database. You MUST then provide all
         microstructure fields (a11, a22, a12, a13, a23, fiber_massfrac, ar).
      3. Fully explicit inputs (no card, no DB lookup): provide all constituent
         properties directly — use when the fiber/polymer is NOT in the database.
         Required: fiber_E1_MPa, fiber_E2_MPa, fiber_G12_MPa, fiber_nu12,
         fiber_nu23, fiber_density_kg_m3, matrix_modulus_MPa, matrix_poisson,
         matrix_density_kg_m3 — PLUS all microstructure fields.

    Any argument set to a value != -1.0 overrides what was loaded from the card or
    datasheet. Use this for what-if scenarios — e.g. change a11 while keeping
    everything else from a saved card.

    Thermoelastic prediction (CTE11, CTE22, CTE33) runs automatically when
    f_cte1_per_K, f_cte2_per_K, and m_cte_per_K are all available — either
    from a card with Stage 2 complete or from explicit override arguments.

    Units: moduli in MPa, CTE in 1/K (NOT ppm/K — multiply ppm/K by 1e-6 first).

    IMPORTANT: This tool does NOT require measured composite properties. It only
    needs constituent inputs and microstructure. Call it directly when the user
    asks for a forward prediction or wants to see predicted composite properties.
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

    elif fiber_E1_MPa > 0:
        # Option C: all constituent properties provided directly — no DB needed
        inputs = {
            "e1":             fiber_E1_MPa,
            "e2":             fiber_E2_MPa   if fiber_E2_MPa   > 0 else None,
            "g12":            fiber_G12_MPa  if fiber_G12_MPa  > 0 else None,
            "f_nu12":         fiber_nu12     if fiber_nu12     > 0 else None,
            "f_nu23":         fiber_nu23     if fiber_nu23     > 0 else None,
            "fiber_density":  fiber_density_kg_m3 if fiber_density_kg_m3 > 0 else None,
            "rho_f":          fiber_density_kg_m3 if fiber_density_kg_m3 > 0 else None,
            "matrix_density": matrix_density_kg_m3 if matrix_density_kg_m3 > 0 else None,
            "rho_m":          matrix_density_kg_m3 if matrix_density_kg_m3 > 0 else None,
            "a12": 0.0, "a13": 0.0, "a23": 0.0,
        }
        inputs = {k: v for k, v in inputs.items() if v is not None}
        source_label = "Explicit constituent inputs (no DB)"

    else:
        return (
            "Specify the material system:\n"
            "  option A — card_id >= 0  (loads fiber + polymer + microstructure from a saved card)\n"
            "  option B — fiber_name='AF' and polymer_name='AP'  "
            "(loads datasheets; also provide a11, a22, fiber_massfrac, ar)\n"
            "  option C — provide fiber_E1_MPa, fiber_E2_MPa, fiber_G12_MPa, fiber_nu12, "
            "fiber_nu23, fiber_density_kg_m3, matrix_modulus_MPa, matrix_poisson, "
            "matrix_density_kg_m3 plus microstructure (a11, a22, fiber_massfrac, ar)"
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

    # ── Orientation tensor positive semi-definiteness check ──────────────────
    _orientation_err = _sinv.validate_orientation_tensor(inputs)
    if _orientation_err:
        return f"Invalid orientation tensor: {_orientation_err}"

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
    p1_WmK: float = -1.0,
    p2_WmK: float = -1.0,
    a11: float = -1.0,
    a22: float = -1.0,
    a12: float = -1.0,
    a13: float = -1.0,
    a23: float = -1.0,
    fiber_massfrac: float = -1.0,
    ar: float = -1.0,
    fiber_density_kg_m3: float = -1.0,
    matrix_density_kg_m3: float = -1.0,
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
      1. card_id >= 0                         — load fiber + polymer + microstructure from a card
      2. fiber_name + polymer_name (strings)  — load from datasheets by name (you must also
                                                provide a11, a22, fiber_massfrac, ar)
      3. Explicit constituent values (fiber_density_kg_m3 > 0, no card or DB name needed):
         provide fiber_density_kg_m3, matrix_density_kg_m3, microstructure (a11, a22,
         fiber_massfrac, ar), and conductivities — use option 3 when the user supplies
         raw numbers without naming a material in the database.

    Constituent conductivity — resolved in priority order:
      1. All three of k_f1_WmK + k_f2_WmK + k_m_WmK — temperature-independent scalar
      2. Parametric: p1_WmK + p2_WmK + k_f1_WmK + k_f2_WmK:
           k_m(T) = p1 * sqrt(T) + p2   (temperature-dependent)
           k_f1, k_f2 are constant.
      3. Stage 3 parametric model stored on the card (p1, p2, k_f1, k_f2).
      4. Scalar k_f1, k_f2, k_m values from the card or datasheet.
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

    elif fiber_density_kg_m3 > 0:
        # Option C: explicit constituent values — no DB lookup needed
        inputs = {"a12": 0.0, "a13": 0.0, "a23": 0.0}
        inputs["rho_f"] = fiber_density_kg_m3
        inputs["fiber_density"] = fiber_density_kg_m3
        if matrix_density_kg_m3 > 0:
            inputs["rho_m"] = matrix_density_kg_m3
            inputs["matrix_density"] = matrix_density_kg_m3
        source_label = "Explicit constituent inputs (no DB)"

    else:
        return (
            "Specify the material system:\n"
            "  option A — card_id >= 0  (loads fiber + polymer + microstructure from a saved card)\n"
            "  option B — fiber_name + polymer_name  (also provide a11, a22, fiber_massfrac, ar)\n"
            "  option C — fiber_density_kg_m3 + matrix_density_kg_m3 + microstructure + k values\n"
            "             (use when supplying raw numbers without a DB material name)"
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

    # ── Orientation tensor positive semi-definiteness check ──────────────────
    _orientation_err = _sinv.validate_orientation_tensor(inputs)
    if _orientation_err:
        return f"Invalid orientation tensor: {_orientation_err}"

    # ── Step 4: Collect k overrides and parametric model inputs ─────────────
    explicit_k: dict = {}
    if k_f1_WmK != -1.0:
        explicit_k["k_f1"] = k_f1_WmK
        inputs["k_f1"] = k_f1_WmK   # also in inputs for parametric model check
        overrides_applied.append(f"  k_f1 = {k_f1_WmK:.4g} W/m·K  [override]")
    if k_f2_WmK != -1.0:
        explicit_k["k_f2"] = k_f2_WmK
        inputs["k_f2"] = k_f2_WmK   # also in inputs for parametric model check
        overrides_applied.append(f"  k_f2 = {k_f2_WmK:.4g} W/m·K  [override]")
    if k_m_WmK != -1.0:
        explicit_k["k_m"] = k_m_WmK
        overrides_applied.append(f"  k_m = {k_m_WmK:.4g} W/m·K  [override]")
    use_explicit_k = len(explicit_k) == 3

    # Parametric k_m model: k_m(T) = p1 * sqrt(T) + p2
    if p1_WmK != -1.0:
        inputs["p1"] = p1_WmK
        overrides_applied.append(f"  p1 = {p1_WmK:.4g} W/m·K  [parametric model]")
    if p2_WmK != -1.0:
        inputs["p2"] = p2_WmK
        overrides_applied.append(f"  p2 = {p2_WmK:.4g} W/m·K  [parametric model]")

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
            "  - Scalar: provide k_f1_WmK, k_f2_WmK, k_m_WmK\n"
            "  - Parametric: provide k_f1_WmK, k_f2_WmK, p1_WmK, p2_WmK\n"
            "    (k_m(T) = p1*sqrt(T) + p2)\n"
            "  - Run Stage 3 (thermal inverse) to infer from k vs T data"
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


# ── Thermal fit comparison tool ───────────────────────────────────────────────

@tool
def compare_thermal_fit(card_id: int, csv_path: str) -> str:
    """
    Compare the card's Stage 3 thermal model predictions against experimental data.

    Loads the inferred constituent thermal properties (p1, p2, k_f1, k_f2) from
    the card, runs the forward surrogate at the experimental temperatures from the
    CSV, and returns a per-channel table of predicted vs. measured values with
    % errors and channel-average RMSE.

    Use this whenever the user asks how the thermal model compares to measurements,
    wants to validate the thermal fit, or asks for predicted vs experimental values.

    card_id : material card with Stage 3 complete
    csv_path: path to CSV file (relative paths resolved against final/ directory).
              Accepted layouts:
                Shared T  — Temperature, K11_WmK, K22_WmK, K33_WmK
                Per-channel T — T_K11, K11_WmK, T_K22, K22_WmK, T_K33, K33_WmK
    """
    import numpy as np

    # ── Load card inputs ──────────────────────────────────────────────────────
    try:
        inputs = _scards.load_card_inputs(card_id)
    except Exception as e:
        return f"Card load error: {e}"

    # Alias normalisation
    for src, dst in [("fiber_density", "rho_f"), ("matrix_density", "rho_m"),
                     ("fiber_massfrac", "w_f"), ("ar", "ar_f")]:
        if dst not in inputs and src in inputs:
            inputs[dst] = inputs[src]
        if src not in inputs and dst in inputs:
            inputs[src] = inputs[dst]

    struct_missing = [f for f in _THERMAL_STRUCTURAL_REQUIRED if f not in inputs]
    if struct_missing:
        return f"Card {card_id} is missing structural inputs: {struct_missing}. Complete Stage 1 first."

    has_parametric = all(inputs.get(p) is not None for p in ("p1", "p2", "k_f1", "k_f2"))
    has_scalar     = all(inputs.get(k) is not None for k in ("k_f1", "k_f2", "k_m"))
    if not (has_parametric or has_scalar):
        return (
            f"Card {card_id} has no thermal conductivity parameters (Stage 3 not complete). "
            "Run run_thermal_inverse first."
        )

    # ── Build ConstituentParams ───────────────────────────────────────────────
    k_f1 = float(inputs["k_f1"])
    k_f2 = float(inputs["k_f2"])
    if has_parametric:
        p1 = float(inputs["p1"])
        p2 = float(inputs["p2"])
    else:
        # Scalar k_m — fit a flat polynomial: p1=0, p2=k_m
        p1 = 0.0
        p2 = float(inputs["k_m"])

    t = k_f1 / k_f2 if k_f2 > 0 else 1.01
    params = _ithermal.ConstituentParams(p1=p1, p2=p2, l2=k_f1, t=t)

    fixed_inputs = {k: float(inputs[k]) for k in _THERMAL_STRUCTURAL_REQUIRED}

    # ── Load CSV ──────────────────────────────────────────────────────────────
    try:
        K_data = _parse_thermal_csv(csv_path)
    except Exception as e:
        return f"CSV error: {e}"

    # ── Load forward model ────────────────────────────────────────────────────
    try:
        from core.services.service_forward import get_model
        predictor = _ithermal.make_batched_predictor(get_model("thermal"))
    except Exception as e:
        return f"Model load error: {e}"

    # ── Evaluate per channel and build comparison table ───────────────────────
    channel_map = [("K11", 0), ("K22", 1), ("K33", 2)]
    lines = [
        f"THERMAL FIT COMPARISON — Card #{card_id}",
        f"Constituent model:  k_f1={k_f1:.4f}  k_f2={k_f2:.4f}  "
        f"p1={p1:.4e}  p2={p2:.4f}  (k_m(T)=p1·√T+p2)",
        "",
    ]

    summary_rows = []
    for ch_key, col_idx in channel_map:
        ch = K_data.get(ch_key)
        if ch is None:
            lines.append(f"{ch_key}: no experimental data in CSV — skipped.")
            continue

        T_exp = ch["T"]
        K_exp = ch["K"]
        K_pred_arr = _ithermal.compute_composite_conductivity(
            params, T_exp, predictor, fixed_inputs
        )
        K_pred = K_pred_arr[:, col_idx]

        residuals  = K_pred - K_exp
        pct_errors = residuals / K_exp * 100.0
        rmse       = float(np.sqrt(np.mean(residuals ** 2)))
        mae_pct    = float(np.mean(np.abs(pct_errors)))

        lines.append(f"{ch_key}  (RMSE={rmse:.4f} W/m·K  |  MAE={mae_pct:.1f}%)")
        lines.append(f"  {'T (°C)':>8}  {'Exp (W/m·K)':>13}  {'Pred (W/m·K)':>13}  {'Error %':>8}")
        lines.append(f"  {'--------':>8}  {'----------':>13}  {'-----------':>13}  {'-------':>8}")
        for i in range(len(T_exp)):
            lines.append(
                f"  {T_exp[i]:>8.1f}  {K_exp[i]:>13.4f}  {K_pred[i]:>13.4f}  {pct_errors[i]:>+8.1f}%"
            )
        lines.append("")
        summary_rows.append((ch_key, rmse, mae_pct))

    if summary_rows:
        lines.append("SUMMARY")
        lines.append(f"  {'Channel':<8}  {'RMSE (W/m·K)':>14}  {'MAE (%)':>8}")
        for ch_key, rmse, mae in summary_rows:
            lines.append(f"  {ch_key:<8}  {rmse:>14.4f}  {mae:>8.1f}%")

    return "\n".join(lines)


# ── Parameter sweep tool ─────────────────────────────────────────────────────

@tool
def sweep_parameter(
    parameter: str,
    values: list[float],
    target_property: str = "E1",
    card_id: int = -1,
    fiber_name: str = "",
    polymer_name: str = "",
    a11: float = -1.0,
    a22: float = -1.0,
    a12: float = 0.0,
    a13: float = 0.0,
    a23: float = 0.0,
    fiber_massfrac: float = -1.0,
    ar: float = -1.0,
    matrix_modulus_MPa: float = -1.0,
    matrix_poisson: float = -1.0,
    fiber_E1_MPa: float = -1.0,
    fiber_E2_MPa: float = -1.0,
    fiber_G12_MPa: float = -1.0,
    fiber_nu12: float = -1.0,
    fiber_nu23: float = -1.0,
    fiber_density_kg_m3: float = -1.0,
    matrix_density_kg_m3: float = -1.0,
) -> str:
    """
    Sweep one microstructure or constituent parameter over a list of values
    and show how a target composite property changes. Use for what-if analysis,
    e.g. "how much do I need to increase fiber_massfrac to reach E1 = 13 GPa?"

    Three ways to specify the base material system — pick ONE:
      1. card_id >= 0: load everything from a saved card.
      2. fiber_name + polymer_name + microstructure fields (a11, a22, fiber_massfrac, ar):
         use when no card exists yet. All microstructure fields not being swept must
         be provided explicitly.
      3. Explicit constituent values (fiber_E1_MPa > 0, no card or DB name needed):
         provide all fiber moduli, densities, and matrix properties directly.
         Microstructure fields (a11, a22, fiber_massfrac, ar) must still be given,
         except for the parameter being swept.

    Parameters
    ----------
    parameter            : ar | a11 | a22 | fiber_massfrac | matrix_modulus | matrix_poisson
    values               : list of values to try, e.g. [0.10, 0.15, 0.20, 0.25]
    target_property      : output to highlight, e.g. "E1", "E2", "G12", "CTE11"
    card_id              : card to base sweep on (option 1)
    fiber_name           : fiber material name (option 2)
    polymer_name         : polymer material name (option 2)
    a11, a22, a12, a13, a23, fiber_massfrac, ar : microstructure (options 2 & 3)
    matrix_modulus_MPa, matrix_poisson           : matrix elastic properties (options 2 & 3)
    fiber_E1_MPa         : fiber axial modulus in MPa — triggers option 3 when > 0
    fiber_E2_MPa         : fiber transverse modulus in MPa
    fiber_G12_MPa        : fiber shear modulus in MPa
    fiber_nu12           : fiber axial Poisson ratio
    fiber_nu23           : fiber transverse Poisson ratio
    fiber_density_kg_m3  : fiber density in kg/m³
    matrix_density_kg_m3 : matrix density in kg/m³
    """
    base_inputs = None

    if card_id < 0:
        if fiber_E1_MPa > 0:
            # Option C — explicit constituent values, no DB lookup
            base_inputs = {
                "e1":   fiber_E1_MPa,
                "a12":  0.0, "a13": 0.0, "a23": 0.0,
            }
            if fiber_E2_MPa        > 0: base_inputs["e2"]            = fiber_E2_MPa
            if fiber_G12_MPa       > 0: base_inputs["g12"]           = fiber_G12_MPa
            if fiber_nu12          > 0: base_inputs["f_nu12"]        = fiber_nu12
            if fiber_nu23          > 0: base_inputs["f_nu23"]        = fiber_nu23
            if fiber_density_kg_m3 > 0:
                base_inputs["rho_f"]        = fiber_density_kg_m3
                base_inputs["fiber_density"] = fiber_density_kg_m3
            if matrix_density_kg_m3 > 0:
                base_inputs["rho_m"]         = matrix_density_kg_m3
                base_inputs["matrix_density"] = matrix_density_kg_m3
            if matrix_modulus_MPa  > 0: base_inputs["matrix_modulus"]  = matrix_modulus_MPa
            if matrix_poisson      > 0: base_inputs["matrix_poisson"]  = matrix_poisson

            micro_overrides = {
                "a11": a11, "a22": a22, "a12": a12, "a13": a13, "a23": a23,
                "fiber_massfrac": fiber_massfrac, "ar": ar,
            }
            for k, v in micro_overrides.items():
                if v != -1.0:
                    base_inputs[k] = v

        elif fiber_name.strip() and polymer_name.strip():
            # Option B — DB lookup by name
            try:
                fiber_id, polymer_id = _resolve_fiber_polymer(fiber_name, polymer_name)
                base_inputs = _smat.get_model_inputs(fiber_id, polymer_id, use_inferred=True)
            except ValueError as e:
                return str(e)
            except Exception as e:
                return f"SWEEP ERROR loading materials: {e}"

            overrides = {
                "a11": a11, "a22": a22, "a12": a12, "a13": a13, "a23": a23,
                "fiber_massfrac": fiber_massfrac, "ar": ar,
                "matrix_modulus": matrix_modulus_MPa, "matrix_poisson": matrix_poisson,
            }
            for k, v in overrides.items():
                if v != -1.0:
                    base_inputs[k] = v

        else:
            return (
                "SWEEP ERROR: provide one of:\n"
                "  option A — card_id >= 0\n"
                "  option B — fiber_name + polymer_name + microstructure\n"
                "  option C — fiber_E1_MPa (and other constituent values) + microstructure"
            )

        # Validate microstructure (excluding the parameter being swept)
        MICRO_FIELDS = {"a11", "a22", "a12", "a13", "a23", "fiber_massfrac", "ar"}
        required = MICRO_FIELDS - {parameter}
        missing_micro = required - set(base_inputs)
        if missing_micro:
            return (
                f"SWEEP ERROR: missing microstructure fields: {sorted(missing_micro)}. "
                f"Re-call with those values set explicitly "
                f"(e.g. a22=0.5 when sweeping a11)."
            )

        if "a11" in base_inputs and "a22" in base_inputs:
            base_inputs["a33"] = 1.0 - base_inputs["a11"] - base_inputs["a22"]

    try:
        result = _sfwd.sweep_parameter(
            parameter=parameter,
            values=values,
            target_property=target_property,
            card_id=card_id,
            base_inputs=base_inputs,
        )
    except Exception as e:
        return f"SWEEP ERROR: {e}"

    prop = result["target_property"]
    rows = result["rows"]

    if card_id >= 0:
        header = f"Card #{card_id}"
    elif fiber_E1_MPa > 0:
        header = "Explicit constituent inputs"
    else:
        header = f"{fiber_name}/{polymer_name}"
    lines = [f"Sweep: {result['parameter']} → {prop}  ({header})\n"]
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


@tool
def update_fiber(
    fiber_id: int,
    name: Optional[str] = None,
    supplier: Optional[str] = None,
    E1_MPa: Optional[float] = None,
    E2_MPa: Optional[float] = None,
    G12_MPa: Optional[float] = None,
    nu12: Optional[float] = None,
    nu23: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    CTE1: Optional[float] = None,
    CTE2: Optional[float] = None,
    k1_WmK: Optional[float] = None,
    k2_WmK: Optional[float] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Edit one or more fields of an existing fiber in the material library.

    fiber_id  — id from list_materials or add_fiber (required)

    All other parameters are optional — only the ones you provide are changed:
      E1_MPa, E2_MPa, G12_MPa — moduli in MPa
      nu12, nu23               — Poisson ratios
      density_kg_m3            — density in kg/m³
      CTE1, CTE2               — axial/transverse CTE in 1/K (datasheet values)
      k1_WmK, k2_WmK          — axial/transverse conductivity in W/m·K (datasheet values)
      notes                    — free-text notes
    """
    field_map = {
        "neat_E1":   E1_MPa,
        "neat_E2":   E2_MPa,
        "neat_G12":  G12_MPa,
        "neat_nu12": nu12,
        "neat_nu23": nu23,
        "neat_rho":  density_kg_m3,
        "neat_CTE1": CTE1,
        "neat_CTE2": CTE2,
        "neat_k1":   k1_WmK,
        "neat_k2":   k2_WmK,
        "neat_notes": notes,
        "name":      name,
        "supplier":  supplier,
    }
    updates = {k: v for k, v in field_map.items() if v is not None}
    if not updates:
        return "No fields specified — nothing to update."
    try:
        _smat.update_fiber(fiber_id, **updates)
    except ValueError as e:
        return f"Cannot update fiber: {e}"
    except Exception as e:
        return f"Database error: {e}"

    changed = ", ".join(f"{k}={v}" for k, v in updates.items())
    return f"Fiber id={fiber_id} updated successfully.\n  Changed: {changed}"


@tool
def update_polymer(
    polymer_id: int,
    name: Optional[str] = None,
    supplier: Optional[str] = None,
    E_MPa: Optional[float] = None,
    nu12: Optional[float] = None,
    density_kg_m3: Optional[float] = None,
    CTE: Optional[float] = None,
    k_WmK: Optional[float] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Edit one or more fields of an existing polymer in the material library.

    polymer_id — id from list_materials or add_polymer (required)

    All other parameters are optional — only the ones you provide are changed:
      E_MPa         — Young's modulus (matrix modulus) in MPa
      nu12          — Poisson ratio
      density_kg_m3 — density in kg/m³
      CTE           — matrix CTE in 1/K (datasheet value)
      k_WmK         — matrix thermal conductivity in W/m·K (datasheet value)
      notes         — free-text notes
    """
    field_map = {
        "neat_E1":  E_MPa,
        "neat_nu12": nu12,
        "neat_rho": density_kg_m3,
        "neat_CTE": CTE,
        "neat_k":   k_WmK,
        "neat_notes": notes,
        "name":     name,
        "supplier": supplier,
    }
    updates = {k: v for k, v in field_map.items() if v is not None}
    if not updates:
        return "No fields specified — nothing to update."
    try:
        _smat.update_polymer(polymer_id, **updates)
    except ValueError as e:
        return f"Cannot update polymer: {e}"
    except Exception as e:
        return f"Database error: {e}"

    changed = ", ".join(f"{k}={v}" for k, v in updates.items())
    return f"Polymer id={polymer_id} updated successfully.\n  Changed: {changed}"


@tool
def add_printer(
    name: str,
    manufacturer: str = "",
) -> str:
    """
    Add a new printer to the database.

    name         — printer name (e.g. "BAAM", "CAMRI-2")
    manufacturer — optional manufacturer name

    Returns the new printer_id. Use this when the user names a printer that is
    not yet in the database before running run_transfer.
    """
    try:
        pid = _smat.add_printer(name=name.strip(), manufacturer=manufacturer.strip())
    except ValueError as e:
        return f"Cannot add printer: {e}"
    except Exception as e:
        return f"Database error: {e}"

    return (
        f"Printer added successfully.\n"
        f"  printer_id   = {pid}\n"
        f"  name         = \"{name}\"\n"
        f"  manufacturer = \"{manufacturer}\"\n"
        f"You can now use \"{name}\" as the target printer in run_transfer."
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
    """Resolve fiber and polymer names to DB IDs. Supports partial/case-insensitive matching."""
    all_fibers   = _db.get_all_fibers()
    all_polymers = _db.get_all_polymers()
    fn = fiber_name.strip().lower()
    pn = polymer_name.strip().lower()

    fiber_matches = [f for f in all_fibers if fn in f["name"].lower()]
    if not fiber_matches:
        raise ValueError(
            f"Fiber '{fiber_name}' not found. Available: {', '.join(f['name'] for f in all_fibers)}"
        )
    if len(fiber_matches) > 1:
        names = ", ".join(f["name"] for f in fiber_matches)
        raise ValueError(f"Fiber '{fiber_name}' matches multiple entries: {names}. Be more specific.")

    polymer_matches = [p for p in all_polymers if pn in p["name"].lower()]
    if not polymer_matches:
        raise ValueError(
            f"Polymer '{polymer_name}' not found. Available: {', '.join(p['name'] for p in all_polymers)}"
        )
    if len(polymer_matches) > 1:
        names = ", ".join(p["name"] for p in polymer_matches)
        raise ValueError(f"Polymer '{polymer_name}' matches multiple entries: {names}. Be more specific.")

    return fiber_matches[0]["id"], polymer_matches[0]["id"]


def _resolve_material_ids(
    fiber_name: str, polymer_name: str, printer_name: str
) -> tuple[int, int, int]:
    """Resolve material names to DB IDs. Supports partial/case-insensitive matching."""
    all_fibers   = _db.get_all_fibers()
    all_polymers = _db.get_all_polymers()
    all_printers = _db.get_all_printers()
    fn = fiber_name.strip().lower()
    pn = polymer_name.strip().lower()
    rn = printer_name.strip().lower()

    fiber_matches = [f for f in all_fibers if fn in f["name"].lower()]
    if not fiber_matches:
        raise ValueError(
            f"Fiber '{fiber_name}' not found. Available: {', '.join(f['name'] for f in all_fibers)}"
        )
    if len(fiber_matches) > 1:
        names = ", ".join(f["name"] for f in fiber_matches)
        raise ValueError(f"Fiber '{fiber_name}' matches multiple entries: {names}. Be more specific.")

    polymer_matches = [p for p in all_polymers if pn in p["name"].lower()]
    if not polymer_matches:
        raise ValueError(
            f"Polymer '{polymer_name}' not found. Available: {', '.join(p['name'] for p in all_polymers)}"
        )
    if len(polymer_matches) > 1:
        names = ", ".join(p["name"] for p in polymer_matches)
        raise ValueError(f"Polymer '{polymer_name}' matches multiple entries: {names}. Be more specific.")

    printer_matches = [p for p in all_printers if rn in p["name"].lower()]
    if not printer_matches:
        raise ValueError(
            f"Printer '{printer_name}' not found. Available: {', '.join(p['name'] for p in all_printers)}"
        )
    if len(printer_matches) > 1:
        names = ", ".join(p["name"] for p in printer_matches)
        raise ValueError(f"Printer '{printer_name}' matches multiple entries: {names}. Be more specific.")

    return fiber_matches[0]["id"], polymer_matches[0]["id"], printer_matches[0]["id"]


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
    # Known constituent properties (pass to fix; omit to let solver infer)
    matrix_modulus_MPa: Optional[float] = None,
    matrix_poisson: Optional[float] = None,
    card_name: str = "",
) -> str:
    """
    Run Stage 1 elastic inverse: infer microstructure (a11, a22, etc.) and
    in-situ constituent properties (matrix_modulus) from measured composite
    elastic properties.

    Pass material NAMES (not IDs) — e.g. fiber_name="AF", polymer_name="AP",
    printer_name="CAMRI". IDs are resolved automatically.

    ORIENTATION (a11, a22):
    - Only pass a11/a22 if the user has explicitly measured them (e.g. from micro-CT
      or a supplier datasheet). Passing them fixes those values and removes them from
      inference.
    - If the user has NOT measured orientation, omit a11 and a22 entirely — the solver
      will infer them from the elastic measurements. Never assume or guess a default
      orientation (e.g. do not set a11=a22=0.333 for "random" unless the user said so).

    MICROSTRUCTURE (ar, fiber_massfrac):
    - If the user provides these (e.g. "mass fraction around 25%"), pass them to fix
      them and reduce the free parameter count.
    - If not provided, omit them — the solver will infer them, though with less certainty.

    CONSTITUENT PROPERTIES (matrix_modulus_MPa, matrix_poisson):
    - Pass matrix_modulus_MPa to fix the matrix stiffness (e.g. if known from neat
      resin testing). Must be a positive value (> 0). If omitted or 0, it is always inferred.
    - Pass matrix_poisson to fix the Poisson ratio regardless of what measurements are
      present. Must be a positive value (> 0). If omitted or 0, it is fixed at the
      datasheet value when no shear/Poisson measurements are available, and freed
      automatically when they are.

    IDENTIFIABILITY:
    - E1+E2 only: a11, a22, ar, and matrix_modulus are all free — the problem is
      underdetermined. Warn the user that results may not be unique, then call once.
    - E1+E2+G12+nu12 or E1+E2+E3: well-constrained. Call without warning.
    - matrix_poisson is only identifiable when at least one shear or Poisson measurement
      is present (G12, G13, G23, nu12, nu13, nu23). Without those it stays fixed at the
      datasheet value automatically — do not try to infer it.

    IDENTIFIABILITY:
    - Run check_identifiability before calling this tool when measurements are sparse.
    - This tool does not run its own FIM — use the separate check_identifiability tool.

    CALL DISCIPLINE:
    - Call this tool exactly once per user request. If the result is poor, report it
      and ask the user how to proceed — do not re-run automatically.

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
    # without them it stays fixed at the datasheet value — unless user explicitly fixes it.
    has_shear = bool(set(targets.keys()) & _SHEAR_POISSON_MEASUREMENTS)
    base_free = _STAGE1_FREE if has_shear else _STAGE1_FREE_WITHOUT_POISSON

    # Free variables = chosen base set minus whatever the user already fixed
    free_vars = [v for v in base_free if v not in known_micro]

    # Apply user-supplied constituent overrides: remove from free list and fix in inputs.
    known_constituent: dict = {}
    if matrix_modulus_MPa is not None and matrix_modulus_MPa > 0:
        known_constituent["matrix_modulus"] = matrix_modulus_MPa
        free_vars = [v for v in free_vars if v != "matrix_modulus"]
    if matrix_poisson is not None and matrix_poisson > 0:
        known_constituent["matrix_poisson"] = matrix_poisson
        free_vars = [v for v in free_vars if v != "matrix_poisson"]

    # Fixed inputs = full datasheet minus the actual free vars + user-provided microstructure
    # + user-provided constituent overrides.
    free_set = set(free_vars)
    fixed_inputs = {k: v for k, v in datasheet.items() if k not in free_set}
    fixed_inputs.update(known_micro)
    fixed_inputs.update(known_constituent)

    # ── Underdetermined check (fast, no JAX) ──────────────────────────────────
    n_meas  = len(targets)
    n_free  = len(free_vars)
    if n_meas < n_free - 2:
        return (
            f"IDENTIFIABILITY WARNING: {n_meas} measurement(s) provided for {n_free} free "
            f"parameter(s) ({', '.join(free_vars)}). The system is underdetermined — "
            "results would be unreliable.\n\n"
            "Recommended minimum measurements: E1 + E2 + E3 (3 measurements).\n"
            "Best: E1 + E2 + E3 + G12 + nu12.\n\n"
            "Run check_identifiability to see which parameters are identifiable with your "
            "current measurement set, or add more measurements before calling this tool."
        )

    # ── Run solver ────────────────────────────────────────────────────────────
    # Strip keys the model doesn't know (e.g. rho_f, rho_m alias fields from DB)
    _model_fields = set(_sfwd.get_input_fields("elastic"))
    fixed_inputs = {k: v for k, v in fixed_inputs.items() if k in _model_fields}

    bounds = {k: _DEFAULT_BOUNDS[k] for k in free_vars if k in _DEFAULT_BOUNDS}

    # ── Identifiability check (automatic, N=30 — lightweight) ────────────────
    _identifiability_warning = ""
    try:
        _fim_result = _sfim.run_fim(
            model_name="elastic",
            fixed_inputs=fixed_inputs,
            free_inputs=free_vars,
            bounds=bounds,
            target_outputs={m: 0.0 for m in targets},
            n_samples=30,
        )
        _poor     = [p for p in free_vars if _fim_result["status"].get(p) == "POOR"]
        _marginal = [p for p in free_vars if _fim_result["status"].get(p) == "MARGINAL"]
        if _poor or _marginal:
            _parts = []
            if _poor:
                _parts.append(f"POOR: {', '.join(_poor)}")
            if _marginal:
                _parts.append(f"MARGINAL: {', '.join(_marginal)}")
            _identifiability_warning = (
                "IDENTIFIABILITY WARNING: The solver ran and results are shown below, "
                "but the available measurements cannot reliably constrain all free parameters "
                f"— {'; '.join(_parts)}. Do NOT call this tool again. "
                "Report the results as-is and warn the user that affected parameters have "
                "high uncertainty. Suggest additional measurements to improve confidence."
            )
    except Exception:
        pass  # FIM failure is non-fatal — solver runs regardless

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

    lines = ["ELASTIC INVERSE — COMPLETE", ""]
    if _identifiability_warning:
        lines += [_identifiability_warning, ""]
    if inv.get("orientation_warning"):
        lines += [
            f"WARNING: {inv['orientation_warning']}",
            "The solved orientation tensor is not physically valid. Do not save this result —",
            "tighten the a11/a22 bounds or fix more microstructure values and re-run.",
            "",
        ]
    lines += [
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
        f"  matrix_modulus  = {known_constituent['matrix_modulus']:.1f} MPa  (fixed — user provided)"
        if "matrix_modulus" in known_constituent else
        f"  matrix_modulus  = {free.get('matrix_modulus', float('nan')):.1f} MPa  (inferred)",
        f"  matrix_poisson  = {known_constituent['matrix_poisson']:.4f}  (fixed — user provided)"
        if "matrix_poisson" in known_constituent else
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
    CTE11_per_K: float,
    CTE22_per_K: float,
    CTE33_per_K: float = -1.0,
    CTE11_sigma_per_K: float = 0.0,
    CTE22_sigma_per_K: float = 0.0,
    CTE33_sigma_per_K: float = 0.0,
    # ── Path A: load Stage 1 from a saved card ────────────────────────────────
    card_id: int = -1,
    # ── Path B: explicit inputs (no saved card needed) ────────────────────────
    fiber_name: str = "",
    polymer_name: str = "",
    printer_name: str = "",
    a11: float = -1.0,
    a22: float = -1.0,
    a12: float = 0.0,
    a13: float = 0.0,
    a23: float = 0.0,
    fiber_massfrac: float = -1.0,
    ar: float = -1.0,
    matrix_modulus_MPa: float = -1.0,
    matrix_poisson: float = -1.0,
) -> str:
    """
    Run Stage 2 thermoelastic inverse: infer fiber CTEs (f_cte1, f_cte2) and
    matrix CTE (m_cte) from measured composite thermal expansion coefficients.

    Two input paths — use exactly one:

    PATH A (card): provide card_id from a completed Stage 1 save. Microstructure
      and matrix properties are loaded from the card automatically.

    PATH B (explicit): provide fiber_name, polymer_name, printer_name plus all
      Stage 1 outputs directly. Use when no card has been saved yet, or to run
      Stage 2 standalone without a prior elastic inverse.
      Required explicit fields: a11, a22, fiber_massfrac, ar,
        matrix_modulus_MPa (in MPa), matrix_poisson.
      a12, a13, a23 default to 0.0 if omitted.

    CTE11_per_K: composite CTE along print direction (1/K). Required.
    CTE22_per_K: composite CTE transverse to print direction (1/K). Required.
    CTE33_per_K: out-of-plane CTE (1/K). Pass -1.0 to exclude.
    CTE sigma fields: 1-sigma uncertainty (1/K). Use 0.0 if unknown.
    Convert ppm/K → 1/K before passing: value × 1e-6.

    Inferred CTEs are constituent (printer-independent) and reusable across
    printers with the same fiber and polymer.
    Results held in memory. Call save_to_card() to persist.
    """
    # ── Resolve inputs: card or explicit ─────────────────────────────────────
    if card_id > 0:
        # Path A: load from saved card
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
        card_name  = cfg.get("name", "")

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

        free_set   = set(_TE_FREE)
        _te_fields = set(_sfwd.get_input_fields("thermoelastic"))
        fixed_inputs = {
            k: v for k, v in all_inputs.items()
            if k not in free_set and k in _te_fields
        }

    elif fiber_name.strip() and polymer_name.strip() and printer_name.strip():
        # Path B: explicit inputs
        missing = []
        if a11 < 0:           missing.append("a11")
        if a22 < 0:           missing.append("a22")
        if fiber_massfrac < 0: missing.append("fiber_massfrac")
        if ar < 0:            missing.append("ar")
        if matrix_modulus_MPa < 0: missing.append("matrix_modulus_MPa")
        if matrix_poisson < 0: missing.append("matrix_poisson")
        if missing:
            return (
                f"Explicit path requires: {', '.join(missing)}. "
                "These are the Stage 1 elastic inverse outputs. "
                "Either run Stage 1 and save a card first (then pass card_id), "
                "or supply all required fields explicitly."
            )

        try:
            fiber_id, polymer_id, printer_id = _resolve_material_ids(
                fiber_name, polymer_name, printer_name
            )
        except ValueError as e:
            return f"Material lookup error: {e}"

        try:
            datasheet = _smat.get_model_inputs(fiber_id, polymer_id)
        except Exception as e:
            return f"Error loading material datasheets: {e}"

        explicit_stage1 = {
            "a11": a11, "a22": a22, "a12": a12, "a13": a13, "a23": a23,
            "fiber_massfrac": fiber_massfrac, "ar": ar,
            "matrix_modulus": matrix_modulus_MPa,
            "matrix_poisson": matrix_poisson,
        }
        free_set   = set(_TE_FREE)
        _te_fields = set(_sfwd.get_input_fields("thermoelastic"))
        fixed_inputs = {k: v for k, v in datasheet.items() if k not in free_set and k in _te_fields}
        fixed_inputs.update({k: v for k, v in explicit_stage1.items() if k not in free_set and k in _te_fields})
        card_name  = ""
        card_id    = -1

    else:
        return (
            "Provide either:\n"
            "  card_id=<N>  (Path A — loads Stage 1 from a saved card)\n"
            "  fiber_name, polymer_name, printer_name + a11, a22, fiber_massfrac, ar, "
            "matrix_modulus_MPa, matrix_poisson  (Path B — explicit Stage 1 inputs)"
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

    # ── Bounds and initial values ─────────────────────────────────────────────
    bounds    = {k: _DEFAULT_BOUNDS[k] for k in _TE_FREE if k in _DEFAULT_BOUNDS}
    init_vals = [_TE_INIT.get(k, 0.0) for k in _TE_FREE]

    # ── Identifiability check (automatic, N=20) ───────────────────────────────
    _identifiability_warning = ""
    try:
        _fim_result = _sfim.run_fim(
            model_name="thermoelastic",
            fixed_inputs=fixed_inputs,
            free_inputs=_TE_FREE,
            bounds=bounds,
            target_outputs={m: 0.0 for m in targets},
            n_samples=20,
        )
        _poor     = [p for p in _TE_FREE if _fim_result["status"].get(p) == "POOR"]
        _marginal = [p for p in _TE_FREE if _fim_result["status"].get(p) == "MARGINAL"]
        if _poor or _marginal:
            _parts = []
            if _poor:
                _parts.append(f"POOR: {', '.join(_poor)}")
            if _marginal:
                _parts.append(f"MARGINAL: {', '.join(_marginal)}")
            _cte33_hint = (
                " Adding CTE33 typically improves separation of f_cte2 and m_cte."
                if "CTE33" not in targets else
                " Even with CTE33, f_cte2 and m_cte are inherently coupled in the composite CTE —"
                " this is a fundamental model limitation, not a missing measurement."
            )
            _identifiability_warning = (
                "IDENTIFIABILITY WARNING: The solver ran and results are shown below, "
                "but the available CTE measurements cannot reliably constrain all free parameters "
                f"— {'; '.join(_parts)}. Do NOT call this tool again. "
                f"Report the results as-is and warn the user.{_cte33_hint}"
            )
    except Exception:
        pass  # FIM failure is non-fatal

    # ── Solver config ─────────────────────────────────────────────────────────
    solver_cfg = dict(_ELASTIC_SOLVER_CFG)
    if sigmas:
        solver_cfg["use_epsilon_loss"] = True

    # ── Multi-start: 5 runs with random init, keep best ──────────────────────
    import random as _random
    _N_STARTS = 5
    _rng = _random.Random(0)
    inv = None
    _best_error = float("inf")

    for _start in range(_N_STARTS):
        if _start == 0:
            _init = init_vals  # first run uses the default midpoint init
        else:
            _init = [
                _rng.uniform(_DEFAULT_BOUNDS[k][0], _DEFAULT_BOUNDS[k][1])
                for k in _TE_FREE
            ]
        try:
            _result = _sinv.run_inverse(
                model_name="thermoelastic",
                fixed_inputs=fixed_inputs,
                free_inputs=_TE_FREE,
                bounds=bounds,
                target_outputs=targets,
                sigmas=sigmas if sigmas else None,
                solver_cfg=solver_cfg,
                init_vals=_init,
            )
            if _result["final_error"] < _best_error:
                _best_error = _result["final_error"]
                inv = _result
        except Exception:
            continue

    if inv is None:
        return "Solver error: all 5 restarts failed."

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
        "card_name":  card_name,
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

    lines = ["THERMOELASTIC INVERSE — COMPLETE", ""]
    if _identifiability_warning:
        lines += [_identifiability_warning, ""]
    lines += [
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
        "Results in memory. Call save_to_card() to persist."
        + (f" Use card_id={card_id}." if card_id > 0 else " Provide a card name to create a new card."),
    ]
    return "\n".join(lines)


# ── CSV helper for run_thermal_inverse ───────────────────────────────────────

def _parse_thermal_csv(csv_path: str):
    """Parse a K vs T CSV. Delegates to service_thermal.load_thermal_data.

    Returns K_data: {"K11": {"T": arr, "K": arr} | None, "K22": ..., "K33": ...}

    Accepted layouts (auto-detected):
      Shared T column : Temperature, K11_WmK, K22_WmK, K33_WmK
      Per-channel T   : T_K11, K11_WmK, T_K22, K22_WmK, T_K33, K33_WmK

    Relative paths are resolved against the final/ project root.
    """
    path = Path(csv_path)
    if not path.is_absolute():
        path = _FINAL / csv_path
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    K_data = _sthermal.load_thermal_data(str(path))
    if K_data.get("K11") is None:
        raise ValueError("K11 column is required but was not found in the CSV.")
    return K_data


@tool
def run_thermal_inverse(
    csv_path: str,
    # ── Path A: load Stage 1 from a saved card ────────────────────────────────
    card_id: int = -1,
    # ── Path B: explicit inputs (no saved card needed) ────────────────────────
    fiber_name: str = "",
    polymer_name: str = "",
    printer_name: str = "",
    a11: float = -1.0,
    a22: float = -1.0,
    a12: float = 0.0,
    a13: float = 0.0,
    a23: float = 0.0,
    fiber_massfrac: float = -1.0,
    ar: float = -1.0,
    n_restarts: int = 100,
) -> str:
    """
    Run Stage 3 thermal inverse: infer fiber conductivities (k_f1 = l2, k_f2 = l2/t)
    and polymer conductivity model parameters (p1, p2) where K_m(T) = p1·√T + p2.

    Two input paths — use exactly one:

    PATH A (card): provide card_id from a completed Stage 1 save. Microstructure
      and material densities are loaded from the card automatically.

    PATH B (explicit): provide fiber_name, polymer_name, printer_name plus Stage 1
      microstructure directly. Use when no card has been saved yet.
      Required: a11, a22, fiber_massfrac, ar.
      Optional: a12, a13, a23 (default 0.0).
      Fiber and polymer densities are loaded from the DB automatically.

    csv_path   : absolute path to CSV file. Required columns:
                   temperature_C (°C) and K11_WmK (W/m·K).
                 Optional: K22_WmK, K33_WmK — improve fit if available.
                 Header row required; column names are case-insensitive.
    n_restarts : optimisation restarts (default 20).

    Results held in memory. Call save_to_card() to persist.
    """
    # ── Resolve inputs: card or explicit ─────────────────────────────────────
    if card_id > 0:
        # Path A: load from saved card
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
        card_name  = cfg.get("name", "")

        try:
            all_inputs = _scards.load_card_inputs(card_id)
        except Exception as e:
            return f"Error loading card inputs: {e}"

        if "matrix_modulus" not in all_inputs or "a11" not in all_inputs:
            return (
                f"Stage 1 (elastic inverse) has not been saved for card id={card_id}. "
                "Run run_elastic_inverse first, save to this card, then run Stage 3."
            )

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

    elif fiber_name.strip() and polymer_name.strip() and printer_name.strip():
        # Path B: explicit inputs
        missing = []
        if a11 < 0:            missing.append("a11")
        if a22 < 0:            missing.append("a22")
        if fiber_massfrac < 0: missing.append("fiber_massfrac")
        if ar < 0:             missing.append("ar")
        if missing:
            return (
                f"Explicit path requires: {', '.join(missing)}. "
                "These are Stage 1 elastic inverse outputs. "
                "Either pass card_id or supply all required microstructure fields."
            )

        try:
            fiber_id, polymer_id, printer_id = _resolve_material_ids(
                fiber_name, polymer_name, printer_name
            )
        except ValueError as e:
            return f"Material lookup error: {e}"

        try:
            datasheet = _smat.get_model_inputs(fiber_id, polymer_id)
        except Exception as e:
            return f"Error loading material datasheets: {e}"

        fixed_inputs = {
            "ar_f":  ar,
            "w_f":   fiber_massfrac,
            "rho_f": datasheet.get("fiber_density"),
            "rho_m": datasheet.get("matrix_density"),
            "a11":   a11,
            "a22":   a22,
            "a12":   a12,
            "a13":   a13,
            "a23":   a23,
        }
        missing = [k for k, v in fixed_inputs.items() if v is None]
        if missing:
            return f"Missing fields from datasheet: {missing}."
        card_name = ""
        card_id   = -1

    else:
        return (
            "Provide either:\n"
            "  card_id=<N>  (Path A — loads Stage 1 from a saved card)\n"
            "  fiber_name, polymer_name, printer_name + a11, a22, fiber_massfrac, ar"
            "  (Path B — explicit Stage 1 inputs)"
        )

    # ── Parse CSV ─────────────────────────────────────────────────────────────
    try:
        K_data = _parse_thermal_csv(csv_path)
    except Exception as e:
        return f"CSV error: {e}"

    n_temps   = sum(ch["T"].shape[0] for ch in K_data.values() if ch is not None)
    k_present = [k for k, v in K_data.items() if v is not None]

    # ── Run inverse estimation ────────────────────────────────────────────────
    try:
        result = _sthermal.run_thermal_inverse(
            fixed_inputs=fixed_inputs,
            K_data=K_data,
            n_restarts=n_restarts,
            seed=0,
        )
    except Exception as e:
        return f"Solver error: {e}"

    # ── Cache for save_to_card ────────────────────────────────────────────────
    _pending_save.clear()
    _pending_save["result"] = dict(result, model="thermal")
    _pending_save["meta"] = {
        "fiber_id":   fiber_id,
        "polymer_id": polymer_id,
        "printer_id": printer_id,
        "card_name":  card_name,
        "card_id":    card_id if card_id != -1 else None,
    }

    # ── Format output ─────────────────────────────────────────────────────────
    p1, p2, l2, t = result["p1"], result["p2"], result["l2"], result["t"]
    best_loss = result["best_loss"]
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

    missing_channels = [c for c in ("K11", "K22", "K33") if c not in k_present]
    if missing_channels:
        lines += [
            "",
            "[WARNING — INCOMPLETE CHANNEL DATA]",
            f"  Missing channels: {', '.join(missing_channels)}.",
            "  All four parameters (k_f1, k_f2, p1, p2) are best recovered when",
            "  K11, K22, and K33 are all available:",
            "    K11 constrains k_f1 (fiber longitudinal conductivity).",
            "    K22 / K33 constrain k_f2 (fiber transverse conductivity).",
            "    All three together constrain p1, p2 (polymer model).",
            f"  With only {', '.join(k_present)}, inferred values — especially",
            "  k_f2 and the p1/p2 split — may be unreliable.",
            "  Recommend: re-measure with all three channels before saving.",
            "  Do NOT call this tool again — report this warning to the user instead.",
        ]

    return "\n".join(lines)


# ── Transfer tool ─────────────────────────────────────────────────────────────

@tool
def run_transfer(
    source_card_id: int,
    new_printer: str,
    E1_MPa: float = -1.0,
    E2_MPa: float = -1.0,
    E3_MPa: float = -1.0,
    G12_MPa: float = -1.0,
    G13_MPa: float = -1.0,
    G23_MPa: float = -1.0,
    nu12: float = -1.0,
    nu13: float = -1.0,
    nu23: float = -1.0,
    E1_sigma_MPa: float = 0.0,
    E2_sigma_MPa: float = 0.0,
    E3_sigma_MPa: float = 0.0,
    G12_sigma_MPa: float = 0.0,
    nu12_sigma: float = 0.0,
    ar: Optional[float] = None,
    a33: Optional[float] = None,
    a12: float = 0.0,
    a13: float = 0.0,
    a23: float = 0.0,
) -> str:
    """Stage 4 — Transfer constituent properties from a fully-characterized source card
    to a new printer by re-running the elastic inverse with constituent properties fixed.

    Free parameters: a11, a22 (orientation in-plane). If ar is not provided, AR is also
    inferred — run check_identifiability first to verify it is identifiable.

    Off-diagonal orientation terms (a12, a13, a23) default to 0.0 (fixed). Override
    only if you have a reason to believe they are non-zero.

    source_card_id : card that is fully characterized (elastic + thermoelastic + thermal).
    new_printer    : name of the target printer (must exist in the database).
    E*_MPa / nu*   : elastic measurements on the new printer (-1.0 to omit).
    ar             : fiber aspect ratio. If known (e.g. from micro-CT), pass it here to
                     fix it. If omitted, AR is inferred from the elastic measurements.
    a33            : expected out-of-plane orientation (a33 = 1 - a11 - a22). Providing
                     this caps a11 + a22 ≤ 1 - a33, preventing the solver from using up
                     all orientation in-plane. Typical LSAM value: 0.10–0.15.
    a12, a13, a23  : off-diagonal orientation tensor components (default 0.0 = fixed).
    """
    # ── Pre-flight 1: source card exists ─────────────────────────────────────
    card = _db.get_print_config(source_card_id)
    if card is None:
        return (
            f"Card {source_card_id} not found. "
            "Use list_cards() to see available cards."
        )
    fiber_id   = card["fiber_id"]
    polymer_id = card["polymer_id"]

    # ── Pre-flight 2: source card must be fully characterized ────────────────
    stages   = _smat.get_completed_stages(source_card_id, fiber_id, polymer_id)
    required = {"elastic", "thermoelastic", "thermal"}
    missing  = required - set(stages)
    if missing:
        return (
            f"Card {source_card_id} is not fully characterized. "
            f"Missing stages: {', '.join(sorted(missing))}. "
            "Complete these before transferring."
        )

    # ── Pre-flight 3: constituent props resolvable ────────────────────────────
    constituent_props = _stransfer.resolve_constituent_props(source_card_id)
    missing_props = [k for k, v in constituent_props.items() if v is None]
    if missing_props:
        return (
            f"Cannot transfer — these constituent properties are missing from "
            f"card {source_card_id}: {', '.join(missing_props)}. "
            "Ensure all three inverse stages completed and saved successfully."
        )

    # ── Pre-flight 4: resolve or create printer ───────────────────────────────
    all_printers = _db.get_all_printers()
    pn = new_printer.strip().lower()
    printer_matches = [p for p in all_printers if pn in p["name"].lower()]
    if len(printer_matches) > 1:
        names = ", ".join(p["name"] for p in printer_matches)
        return f"Printer '{new_printer}' matches multiple entries: {names}. Be more specific."
    if not printer_matches:
        # Printer doesn't exist — create it automatically.
        # The user named it explicitly, which is implicit authorization.
        target_printer_id = _db.add_printer(new_printer.strip(), manufacturer="")
        _printer_created = True
    else:
        target_printer_id = printer_matches[0]["id"]
        _printer_created = False

    # ── Build target outputs and sigmas ──────────────────────────────────────
    _elastic_map = {
        "E1": (E1_MPa, E1_sigma_MPa), "E2": (E2_MPa, E2_sigma_MPa),
        "E3": (E3_MPa, E3_sigma_MPa), "G12": (G12_MPa, G12_sigma_MPa),
        "G13": (G13_MPa, 0.0),        "G23": (G23_MPa, 0.0),
        "nu12": (nu12, nu12_sigma),   "nu13": (nu13, 0.0), "nu23": (nu23, 0.0),
    }
    target_outputs: dict[str, float] = {}
    sigmas: dict[str, float]         = {}
    for prop, (val, sig) in _elastic_map.items():
        if val is not None and val > 0:
            target_outputs[prop] = float(val)
            if sig > 0:
                sigmas[prop] = float(sig)

    if not target_outputs:
        return (
            "No elastic measurements provided. "
            "Supply at least E1_MPa, E2_MPa, or G12_MPa for the new printer."
        )

    # ── Fixed microstructure: mass fraction always fixed from source card ─────
    fixed_micro: dict[str, float] = {}
    snap = _db.get_latest_microstructure(source_card_id)
    if snap:
        _mf = snap.get("fiber_massfrac") or snap.get("mf")
        if _mf is not None:
            fixed_micro["fiber_massfrac"] = float(_mf)

    if ar is not None:
        fixed_micro["ar"] = float(ar)

    # Off-diagonal orientation terms are fixed (default 0.0 — physically reasonable
    # for printed composites; user can override via a12/a13/a23 args).
    fixed_micro["a12"] = float(a12)
    fixed_micro["a13"] = float(a13)
    fixed_micro["a23"] = float(a23)

    # a33 caps the in-plane sum: tighten the upper bound on a11 + a22.
    # service_transfer uses per-field bounds; cap both a11 and a22 so their
    # maximum combined value cannot exceed 1 - a33.
    extra_bounds: dict[str, tuple[float, float]] | None = None
    if a33 is not None:
        in_plane_max = max(0.01, 1.0 - float(a33))
        extra_bounds = {
            "a11": (0.0, in_plane_max),
            "a22": (0.0, in_plane_max),
        }

    # ── Run transfer solver ───────────────────────────────────────────────────
    try:
        result = _stransfer.run_transfer(
            source_card_id=source_card_id,
            target_outputs=target_outputs,
            sigmas=sigmas or None,
            fixed_micro=fixed_micro,
            bounds=extra_bounds,
        )
    except Exception as e:
        return f"Transfer solver error: {e}"

    # ── Cache for save_to_card ────────────────────────────────────────────────
    _pending_save.clear()
    _pending_save["result"] = dict(result, model="transfer")
    _pending_save["meta"]   = {
        "fiber_id":         fiber_id,
        "polymer_id":       polymer_id,
        "source_card_id":   source_card_id,
        "target_printer_id": target_printer_id,
        "new_printer":      new_printer.strip(),
    }

    # ── Format output ─────────────────────────────────────────────────────────
    micro = result["opt_microstructure"]
    lines = [
        f"TRANSFER RESULT — card {source_card_id} → {new_printer}",
    ]
    if _printer_created:
        lines.append(f"  (printer '{new_printer}' was not in the database — added automatically)")
    lines += [
        "",
        "CONSTITUENT PROPERTIES (carried over):",
    ]
    for key, info in constituent_props.items():
        if info:
            tag = info.get("source_tag", "")
            lines.append(f"  {key:<18s} = {info['value']:.4g}  [{tag}]")

    ar_val    = micro.get("ar") or micro.get("ar_f") or fixed_micro.get("ar")
    ar_status = "fixed" if ar is not None else "inferred"
    mf_val    = fixed_micro.get("fiber_massfrac", "—")

    _a11 = micro.get("a11", 0.0)
    _a22 = micro.get("a22", 0.0)
    _a33 = 1.0 - _a11 - _a22

    lines += [
        "",
        f"INFERRED MICROSTRUCTURE ({new_printer}):",
        f"  a11              = {_a11:.4f}  (inferred)",
        f"  a22              = {_a22:.4f}  (inferred)",
        f"  a33              = {_a33:.4f}  (= 1 - a11 - a22)" + (f"  [target: {a33:.4f}]" if a33 is not None else ""),
        f"  a12              = {fixed_micro['a12']:.4f}  (fixed)",
        f"  a13              = {fixed_micro['a13']:.4f}  (fixed)",
        f"  a23              = {fixed_micro['a23']:.4f}  (fixed)",
        f"  ar               = {ar_val:.2f}  ({ar_status})" if ar_val else f"  ar               = — ({ar_status})",
        f"  fiber_massfrac   = {mf_val:.4f}  (fixed from source card)" if isinstance(mf_val, float) else f"  fiber_massfrac   = {mf_val}",
        "",
        f"Elastic fit error  = {result['elastic_error']:.5f}",
        "",
        "Call save_to_card(card_name='...') to create the new card.",
    ]
    return "\n".join(lines)


@tool
def run_full_pipeline(
    file_path: str,
    fiber_name: str,
    polymer_name: str,
    printer_name: str,
    fiber_massfrac: float = -1.0,
    aspect_ratio: float = -1.0,
    a11: float = -1.0,
    a22: float = -1.0,
    n_restarts: int = 20,
) -> str:
    """
    Run all 3 inverse stages (elastic → thermoelastic → thermal) from a single
    Excel measurements file. Results are held in memory — call save_to_card()
    with a card name after reviewing the results.

    Use this when the user provides a file path (.xlsx) containing experimental
    measurements. Do NOT use if the user is providing measurements directly in
    the conversation — use run_elastic_inverse instead.

    The .xlsx file must have:
      - Sheet 'measurements': key-value table (column A = field name, column B = value).
        Elastic fields: E1_MPa, E2_MPa, E3_MPa, G12_MPa, G13_MPa, G23_MPa, nu12, nu13, nu23
        Uncertainty:    E1_sigma_MPa, E2_sigma_MPa, ... nu12_sigma, nu13_sigma, nu23_sigma
        CTE fields:     CTE11_per_K, CTE22_per_K, CTE33_per_K (optional)
        CTE sigma:      CTE11_sigma_per_K, CTE22_sigma_per_K, CTE33_sigma_per_K
        Omit any field you don't have — do NOT put fiber/polymer/printer names in this sheet.
      - Sheet 'thermal' (optional): columns temperature_c and K11_WmK (+ optional K22_WmK, K33_WmK)
        If absent, Stage 3 is skipped.

    Material identity (fiber_name, polymer_name, printer_name) must be confirmed
    in the database before calling this — use list_materials() first, and
    add_fiber()/add_polymer() if needed.

    Microstructure params (fiber_massfrac, aspect_ratio, a11, a22) are optional —
    pass values only if the user provided them from CT or process data.
    Pass -1.0 (default) to let the solver infer them.

    After this tool returns, ask the user for a card name, then call
    save_to_card(card_name='...') to persist all results with correct provenance.
    """
    import core.services.service_pipeline as _spipeline

    # Resolve names to IDs
    try:
        fiber_id, polymer_id, printer_id = _resolve_material_ids(
            fiber_name, polymer_name, printer_name
        )
    except ValueError as e:
        return f"Material lookup error: {e}"

    result = _spipeline.run_pipeline(
        file_path=file_path,
        fiber_id=fiber_id,
        polymer_id=polymer_id,
        printer_id=printer_id,
        fiber_massfrac=fiber_massfrac,
        aspect_ratio=aspect_ratio,
        a11=a11,
        a22=a22,
        n_restarts=n_restarts,
    )

    # ── Store for save_to_card ────────────────────────────────────────────────
    _pending_pipeline.clear()
    _pending_pipeline.update(result)
    _pending_pipeline["meta"] = {
        "fiber_id":   fiber_id,
        "polymer_id": polymer_id,
        "printer_id": printer_id,
        "fiber_name": fiber_name,
        "polymer_name": polymer_name,
        "printer_name": printer_name,
    }

    # ── Format report ─────────────────────────────────────────────────────────
    def _stage_header(n: int, label: str, stage: dict) -> str:
        status = stage["status"].upper()
        err    = stage.get("fit_error")
        err_str = f"  fit_error = {err:.5f}" if err is not None else ""
        return f"STAGE {n} — {label}: {status}{err_str}"

    def _fmt_cte(v: float) -> str:
        return f"{v * 1e6:.4f} ppm/K"

    s1 = result["stage1"]
    s2 = result["stage2"]
    s3 = result["stage3"]

    lines = ["=== BATCH PIPELINE REPORT ===", ""]

    # Stage 1
    lines.append(_stage_header(1, "Elastic Inverse", s1))
    if s1["status"] == "pass":
        inf = s1["inferred"]
        lines += [
            f"  matrix_modulus  = {inf.get('matrix_modulus', float('nan')):.1f} MPa",
            f"  matrix_poisson  = {inf.get('matrix_poisson', float('nan')):.4f}",
            f"  a11             = {inf.get('a11', float('nan')):.4f}",
            f"  a22             = {inf.get('a22', float('nan')):.4f}",
            f"  fiber_massfrac  = {inf.get('fiber_massfrac', float('nan')):.4f}",
            f"  ar              = {inf.get('ar', float('nan')):.2f}",
        ]
    elif s1.get("note"):
        lines.append(f"  {s1['note']}")
    lines.append("")

    # Stage 2
    lines.append(_stage_header(2, "Thermoelastic Inverse", s2))
    if s2["status"] == "pass":
        inf = s2["inferred"]
        lines += [
            f"  f_cte1 (fiber axial)      = {_fmt_cte(inf.get('f_cte1', float('nan')))}",
            f"  f_cte2 (fiber transverse) = {_fmt_cte(inf.get('f_cte2', float('nan')))}",
            f"  m_cte  (matrix)           = {_fmt_cte(inf.get('m_cte',  float('nan')))}",
        ]
    elif s2.get("note"):
        lines.append(f"  {s2['note']}")
    lines.append("")

    # Stage 3
    lines.append(_stage_header(3, "Thermal Inverse", s3))
    if s3["status"] == "pass":
        inf = s3["inferred"]
        k_m25 = inf["p1"] * (25.0 ** 0.5) + inf["p2"]
        lines += [
            f"  k_f1 (fiber longitudinal) = {inf.get('k_f1', float('nan')):.4f} W/m·K",
            f"  k_f2 (fiber transverse)   = {inf.get('k_f2', float('nan')):.4f} W/m·K",
            f"  p1 (polymer scaling)      = {inf.get('p1', float('nan')):.4e} W/m·K",
            f"  p2 (polymer offset)       = {inf.get('p2', float('nan')):.4f} W/m·K",
            f"  k_m @ 25°C               = {k_m25:.4f} W/m·K",
        ]
    elif s3.get("note"):
        lines.append(f"  {s3['note']}")
    lines.append("")

    # Errors
    if result["errors"]:
        lines += ["ERRORS:"]
        for e in result["errors"]:
            lines.append(f"  • {e}")
        lines.append("")

    # Always end with the save prompt
    ran = [f"Stage {i+1}" for i, s in enumerate([s1, s2, s3]) if s["status"] == "pass"]
    lines.append(
        f"Results in memory ({', '.join(ran)} complete). "
        "Call save_to_card(card_name='...') with a name of your choice to save all results."
    )

    return "\n".join(lines)


@tool
def save_to_card(card_name: str = "", card_id: int = -1) -> str:
    """
    Save the most recent solver result (or full pipeline result) to the database.

    card_name: name for the new card (e.g. "T300/PESU CAMRI run1"). ALWAYS ask
               the user for a card name before calling this tool if card_id=-1.
    card_id = -1  → create a new material card using card_name
    card_id >= 0  → save to an existing card (updates it in place, card_name ignored)

    Only call this after:
      1. A solver tool (run_elastic_inverse, run_full_pipeline, etc.) returned a
         successful result, AND
      2. The user has explicitly confirmed they want to save, AND
      3. You have asked for and received a card_name (when card_id=-1).

    Do NOT call automatically — always ask the user for a card name first.
    Returns the card_id so subsequent tools can reference it.
    """
    # ── Pipeline save: run_full_pipeline stored results in _pending_pipeline ───
    if _pending_pipeline:
        name = card_name.strip()
        if not name:
            return "Please provide a card_name to save the pipeline results."

        meta = _pending_pipeline.get("meta", {})
        s1   = _pending_pipeline.get("stage1", {})
        s2   = _pending_pipeline.get("stage2", {})
        s3   = _pending_pipeline.get("stage3", {})

        if s1.get("status") != "pass" or s1.get("save_data") is None:
            return (
                "Pipeline Stage 1 did not complete successfully — nothing to save. "
                "Check the pipeline report for errors."
            )

        saved_stages = []
        try:
            # Stage 1 — creates the card and gets card_id
            sd1 = s1["save_data"]
            saved_id = _scards.save_inverse_result(
                result=sd1["result"],
                fiber_id=sd1["fiber_id"],
                polymer_id=sd1["polymer_id"],
                printer_id=sd1["printer_id"],
                card_id=None,
                card_name=name,
            )
            saved_stages.append("Stage 1 (elastic)")

            # Stage 2 — save thermoelastic to the same card
            if s2.get("status") == "pass" and s2.get("save_data") is not None:
                sd2 = s2["save_data"]
                _scards.save_inverse_result(
                    result=sd2["result"],
                    fiber_id=sd2["fiber_id"],
                    polymer_id=sd2["polymer_id"],
                    printer_id=sd2["printer_id"],
                    card_id=saved_id,
                    card_name=name,
                )
                saved_stages.append("Stage 2 (thermoelastic)")

            # Stage 3 — save thermal to the same card
            if s3.get("status") == "pass" and s3.get("save_data") is not None:
                sd3 = s3["save_data"]
                _scards.save_thermal_result(
                    result=sd3["result"],
                    fiber_id=sd3["fiber_id"],
                    polymer_id=sd3["polymer_id"],
                    card_id=saved_id,
                )
                saved_stages.append("Stage 3 (thermal)")

        except Exception as e:
            return f"Save error during pipeline save: {e}"

        _pending_pipeline.clear()
        return (
            f"Pipeline saved successfully.\n"
            f"  card_id    = {saved_id}\n"
            f"  card_name  = {name}\n"
            f"  Stages saved: {', '.join(saved_stages)}\n"
            f"Use card_id={saved_id} in get_card_status, predict_properties, "
            f"or predict_thermal_conductivity."
        )

    # ── Single-stage save: run_elastic_inverse / run_thermal_inverse ──────────
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
        if result.get("model") == "transfer":
            target_printer_id = meta.get("target_printer_id")
            source_card_id    = meta.get("source_card_id")
            saved_id = _scards.save_transfer_result(
                result=result,
                source_card_id=source_card_id,
                target_printer_id=target_printer_id,
            )
            _pending_save.clear()
            return (
                f"Transfer saved successfully.\n"
                f"  card_id    = {saved_id}\n"
                f"  card_name  = {name}\n"
                f"  source     = card {source_card_id}\n"
                f"  printer    = {meta.get('new_printer', '')}\n"
                f"Use card_id={saved_id} in predict_properties or get_card_status."
            )

        if result.get("model") == "thermal":
            # Fall back to the card_id captured during run_thermal_inverse
            if card_id == -1:
                card_id = meta.get("card_id") or -1
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
def delete_card(card_id: int, confirm: bool = False) -> str:
    """
    Delete a material card and all data scoped to it from the database.

    Deletes: microstructure snapshots, inference runs, experimental measurements,
    composite property values, thermal predictions, property preferences, and
    printing conditions — everything tied to this card_id.

    Does NOT delete constituent properties (matrix_modulus, f_cte1, k_f1, etc.)
    because they are stored globally against the fiber/polymer pair and may be
    shared with other cards that use the same materials.

    confirm=False (default): performs a dry run — shows exactly what would be
    deleted without making any changes. Always call with confirm=False first.

    confirm=True: performs the actual deletion. Only call after showing the user
    the dry-run summary and receiving explicit confirmation.

    card_id: the card_id to delete (from list_cards or get_card_status).
    """
    try:
        result = _db.delete_card(card_id, dry_run=not confirm)
    except ValueError as e:
        return f"Card not found: {e}"
    except Exception as e:
        return f"Delete error: {e}"

    name = result["card_name"]

    if result["dry_run"]:
        would = result["would_delete"]
        cpv   = result["would_null_cpv_run_ids"]
        lines = [
            f"DRY RUN — nothing deleted yet.",
            f"Card: \"{name}\"  (card_id={card_id})",
            "",
            "Would delete:",
        ]
        for table, count in would.items():
            if count:
                lines.append(f"  {table:<36} {count} row{'s' if count != 1 else ''}")
        if cpv:
            lines.append(
                f"\n  constituent_property_values: {cpv} row{'s' if cpv != 1 else ''} "
                f"will have inference_run_id set to NULL\n"
                f"  (values and provenance tags are preserved — only the audit trail link is broken)"
            )
        lines += [
            "",
            "Constituent properties (matrix_modulus, f_cte1, k_f1, …) are NOT deleted",
            "— they are global to the fiber/polymer pair.",
            "",
            f"To confirm deletion, call delete_card(card_id={card_id}, confirm=True).",
        ]
        return "\n".join(lines)

    # Confirmed deletion
    deleted = result["rows_deleted"]
    cpv     = result["constituent_property_values_inference_run_id_nulled"]
    total   = sum(deleted.values())
    lines   = [
        f"Card \"{name}\" (card_id={card_id}) deleted.",
        f"Total rows removed: {total}",
    ]
    for table, count in deleted.items():
        if count:
            lines.append(f"  {table:<36} {count} row{'s' if count != 1 else ''} deleted")
    if cpv:
        lines.append(
            f"  constituent_property_values: {cpv} row{'s' if cpv != 1 else ''} "
            f"had inference_run_id set to NULL"
        )
    return "\n".join(lines)


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


# ── System administration tools ──────────────────────────────────────────────

@tool
def reinitialize_knowledge_base(full_reset: bool = False) -> str:
    """
    Rebuild the knowledge base from PDFs in agent/knowledge/.

    Use this when:
      - The user adds new PDF papers or datasheets to agent/knowledge/ and wants
        the agent to be able to search them.
      - The user says the knowledge base is missing a paper they added.
      - The knowledge base returns stale or incorrect results.

    full_reset=False (default): incremental — adds any new PDFs not yet indexed,
      skips PDFs that are already in the vector store. Safe to run at any time.

    full_reset=True: wipes the entire vector store (.chroma and .parents) and
      re-ingests all PDFs from scratch. Use this if the embeddings seem wrong or
      the user wants a clean rebuild. Takes longer than incremental.

    Lists PDFs currently in agent/knowledge/ before and after ingestion.
    """
    import shutil
    import agent.rag as _rag
    from agent.ingest import ingest, KNOWLEDGE_DIR, CHROMA_DIR, PARENTS_DIR

    pdfs = sorted(KNOWLEDGE_DIR.glob("*.pdf"))
    pdf_names = [p.name for p in pdfs]

    if not pdfs:
        return (
            f"No PDF files found in agent/knowledge/.\n"
            f"Add PDFs to that directory, then call reinitialize_knowledge_base() again."
        )

    if full_reset:
        removed = []
        if CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
            removed.append(".chroma (vector store)")
        if PARENTS_DIR.exists():
            shutil.rmtree(PARENTS_DIR)
            removed.append(".parents (full-page cache)")
        if removed:
            reset_note = f"Wiped: {', '.join(removed)}\n"
        else:
            reset_note = "Nothing to wipe — starting fresh.\n"
    else:
        reset_note = ""

    # Invalidate the lazy singleton so the next search picks up the new index
    _rag._retriever = None

    try:
        ingest()
    except Exception as e:
        return f"Ingest error: {e}"

    lines = [
        f"{'Full rebuild' if full_reset else 'Incremental update'} complete.",
        reset_note.strip(),
        f"PDFs indexed ({len(pdfs)}):",
    ]
    for name in pdf_names:
        lines.append(f"  • {name}")
    lines.append(
        "\nThe knowledge base is ready. "
        "Use search_knowledge_base() to query it."
    )
    return "\n".join(l for l in lines if l)


@tool
def reset_material_database(keep_library: bool = True, confirm: bool = False) -> str:
    """
    Reset the material database.

    Two modes — choose based on what the user wants to keep:

    keep_library=True (default): clears all characterization results
      (material cards, microstructure snapshots, inference runs, constituent
      property values, composite predictions, experimental measurements) but
      keeps the fiber, polymer, and printer entries. Use this when the user
      wants to redo characterization from scratch without losing their
      material library.

    keep_library=False: full factory reset — drops every table and reseeds
      from the JSON seed files (data/fibers.json, data/polymers.json).
      WARNING: any fibers, polymers, or printers the user added manually
      via add_fiber() / add_polymer() will be permanently lost.

    confirm=False (default): dry run — describes exactly what would be
      deleted without making any changes. Always show this first.

    confirm=True: performs the actual reset. Only call after the user has
      explicitly confirmed after seeing the dry-run output.
    """
    import db.init_db as _init_db

    if not confirm:
        # Dry run — count rows without touching anything
        with _db._connect() as conn:
            def _count(table):
                return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

            card_tables = {
                "print_configs":               _count("print_configs"),
                "microstructure_snapshots":    _count("microstructure_snapshots"),
                "inference_runs":              _count("inference_runs"),
                "constituent_property_values": _count("constituent_property_values"),
                "experimental_measurements":   _count("experimental_measurements"),
                "composite_property_values":   _count("composite_property_values"),
                "current_composite_properties":_count("current_composite_properties"),
                "thermal_k_predictions":       _count("thermal_k_predictions"),
                "property_preferences":        _count("property_preferences"),
                "processing_conditions":       _count("processing_conditions"),
            }
            library = {
                "fibers":   _count("fibers"),
                "polymers": _count("polymers"),
                "printers": _count("printers"),
            }

        lines = [
            "DRY RUN — nothing deleted yet.",
            "",
            f"Mode: {'clear cards only (keep fiber/polymer/printer library)' if keep_library else 'full factory reset (library will be reseeded from JSON)'}",
            "",
            "Would delete (characterization data):",
        ]
        for tbl, n in card_tables.items():
            if n:
                lines.append(f"  {tbl:<36} {n} row{'s' if n != 1 else ''}")

        if not keep_library:
            lines += ["", "Would also wipe and reseed:"]
            for tbl, n in library.items():
                lines.append(f"  {tbl:<36} {n} row{'s' if n != 1 else ''} → replaced by seed JSON")
        else:
            lines += ["", "Would keep (library):"]
            for tbl, n in library.items():
                lines.append(f"  {tbl:<36} {n} row{'s' if n != 1 else ''} preserved")

        lines += [
            "",
            f"To confirm, call reset_material_database(keep_library={keep_library}, confirm=True).",
        ]
        return "\n".join(lines)

    # ── Confirmed reset ────────────────────────────────────────────────────────
    if keep_library:
        _CARD_TABLES = (
            "thermal_k_predictions",
            "current_composite_properties",
            "property_preferences",
            "composite_property_values",
            "experimental_measurements",
            "constituent_property_values",
            "microstructure_snapshots",
            "inference_runs",
            "print_configs",
            "processing_conditions",
        )
        with _db._connect() as conn:
            conn.execute("PRAGMA foreign_keys=OFF")
            for tbl in _CARD_TABLES:
                conn.execute(f"DELETE FROM {tbl}")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.commit()
        return (
            "Characterization data cleared.\n"
            "All material cards, inference runs, microstructure snapshots, "
            "and experimental measurements have been deleted.\n"
            "Fiber, polymer, and printer library is intact.\n"
            "You can now start fresh characterization with run_elastic_inverse "
            "or run_full_pipeline."
        )
    else:
        try:
            _init_db.init(reset=True)
        except Exception as e:
            return f"Full reset error: {e}"
        return (
            "Full factory reset complete.\n"
            "All tables dropped and recreated. "
            "Fibers, polymers seeded from data/fibers.json and data/polymers.json.\n"
            "Any materials you added manually (add_fiber, add_polymer) have been removed.\n"
            "You can verify the library with list_materials()."
        )


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
    compare_thermal_fit,
    add_fiber,
    add_polymer,
    update_fiber,
    update_polymer,
    add_printer,
    check_identifiability,
    sweep_parameter,
    run_elastic_inverse,
    run_thermoelastic_inverse,
    run_thermal_inverse,
    run_transfer,
    run_full_pipeline,
    save_to_card,
    delete_card,
    save_processing_conditions,
    reinitialize_knowledge_base,
    reset_material_database,
]
