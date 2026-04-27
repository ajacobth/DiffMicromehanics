"""service_material.py — material library queries (fibers, polymers, printers, cards)."""
from __future__ import annotations

import db.db as _db


def list_fibers() -> list[dict]:
    """[{id, name, supplier, ...}, ...]"""
    return _db.get_all_fibers()


def list_polymers() -> list[dict]:
    """[{id, name, supplier, ...}, ...]"""
    return _db.get_all_polymers()


def list_printers() -> list[dict]:
    """[{id, name, manufacturer}, ...]"""
    return _db.get_all_printers()


def list_cards(printer_id: int | None = None) -> list[dict]:
    """[{id, name, fiber_id, polymer_id, printer_id, created_at}, ...]
    If printer_id given, filters to that printer.
    """
    if printer_id is not None:
        return _db.get_print_configs_for_printer(printer_id)
    return _db.get_all_print_configs()


def get_fiber_inputs(fiber_id: int) -> dict[str, float]:
    """Model-unit inputs from datasheet (e1, e2, g12, f_nu12, f_nu23, fiber_density, rho_f)."""
    return _db.fiber_model_inputs(fiber_id)


def get_polymer_inputs(polymer_id: int) -> dict[str, float]:
    """Model-unit inputs from datasheet (matrix_modulus, matrix_poisson, matrix_density, rho_m)."""
    return _db.polymer_model_inputs(polymer_id)


def get_inferred_inputs(fiber_id: int, polymer_id: int) -> dict[str, float]:
    """Most-recent inferred constituent properties for this (fiber, polymer) pair.

    Returns {} if nothing has been inferred yet.
    Merges fiber and polymer constituent_property_values where source_tag='inferred'.
    """
    result: dict[str, float] = {}

    for ctype, cid in (("fiber", fiber_id), ("polymer", polymer_id)):
        seen: set[str] = set()
        for p in _db.get_constituent_properties(ctype, cid, include_global=True):
            name = p["property_name"]
            if p["source_tag"] == "inferred" and name not in seen:
                result[name] = float(p["value"])
                seen.add(name)

    return result


def has_inferred_data(fiber_id: int, polymer_id: int) -> bool:
    """True if any inferred constituent_property_value exists for this (fiber, polymer) pair."""
    for ctype, cid in (("fiber", fiber_id), ("polymer", polymer_id)):
        for p in _db.get_constituent_properties(ctype, cid, include_global=True):
            if p["source_tag"] == "inferred":
                return True
    return False


def get_completed_stages(card_id: int, fiber_id: int, polymer_id: int) -> list[str]:
    """Return which characterization stages are complete for a card.

    Checks constituent_property_values for the sentinel inferred properties
    that each stage writes:
        elastic       → polymer inferred matrix_modulus
        thermoelastic → fiber inferred f_cte1
        thermal       → fiber inferred k_f1

    Returns a list in order, e.g. ["elastic", "thermoelastic"].
    """
    stages = []

    def _has_inferred(ctype: str, cid: int, prop: str) -> bool:
        rows = _db.get_constituent_properties(
            ctype, cid, property_name=prop,
            print_config_id=card_id, include_global=True,
        )
        return any(r["source_tag"] == "inferred" for r in rows)

    if _has_inferred("polymer", polymer_id, "matrix_modulus"):
        stages.append("elastic")
    if _has_inferred("fiber", fiber_id, "f_cte1"):
        stages.append("thermoelastic")
    if _has_inferred("fiber", fiber_id, "k_f1"):
        stages.append("thermal")

    return stages


def add_fiber(
    name: str,
    supplier: str,
    E1: float,
    E2: float,
    G12: float,
    nu12: float,
    nu23: float,
    rho: float,
    notes: str = "",
) -> int:
    """Add a new fiber to the material library. Returns the new fiber id.

    All modulus values in MPa, density in kg/m³.
    Raises ValueError if name is empty or already exists.
    """
    name = name.strip()
    if not name:
        raise ValueError("Fiber name is required.")
    existing = {f["name"] for f in _db.get_all_fibers()}
    if name in existing:
        raise ValueError(f"A fiber named '{name}' already exists.")
    return _db.add_fiber(
        name=name,
        supplier=supplier.strip(),
        neat={"E1": E1, "E2": E2, "G12": G12, "nu12": nu12, "nu23": nu23,
              "rho": rho, "notes": notes},
    )


def add_polymer(
    name: str,
    supplier: str,
    E1: float,
    nu12: float,
    rho: float,
    notes: str = "",
) -> int:
    """Add a new polymer to the material library. Returns the new polymer id.

    E1 in MPa (= matrix modulus), density in kg/m³.
    Raises ValueError if name is empty or already exists.
    """
    name = name.strip()
    if not name:
        raise ValueError("Polymer name is required.")
    existing = {p["name"] for p in _db.get_all_polymers()}
    if name in existing:
        raise ValueError(f"A polymer named '{name}' already exists.")
    return _db.add_polymer(
        name=name,
        supplier=supplier.strip(),
        neat={"E1": E1, "nu12": nu12, "rho": rho, "notes": notes},
    )


def add_printer(name: str, manufacturer: str = "") -> int:
    """Add a new printer to the registry. Returns the new printer id.

    Raises ValueError if name is empty or already exists.
    """
    name = name.strip()
    if not name:
        raise ValueError("Printer name is required.")
    existing = {p["name"] for p in _db.get_all_printers()}
    if name in existing:
        raise ValueError(f"A printer named '{name}' already exists.")
    return _db.add_printer(name=name, manufacturer=manufacturer.strip())


def _get_densities(fiber_id: int, polymer_id: int) -> tuple[float, float]:
    """Return (rho_f, rho_m) from DB. Raises ValueError if either is missing."""
    rho_f = get_fiber_inputs(fiber_id).get("fiber_density")
    rho_m = get_polymer_inputs(polymer_id).get("matrix_density")
    if rho_f is None:
        raise ValueError(f"Fiber id={fiber_id} has no density in the database.")
    if rho_m is None:
        raise ValueError(f"Polymer id={polymer_id} has no density in the database.")
    return rho_f, rho_m


def convert_mass_to_volume_fraction(
    fiber_id: int,
    polymer_id: int,
    mass_fraction: float,
) -> dict:
    """Convert fiber mass fraction (wf) → volume fraction (Vf).

    Formula: Vf = (wf · ρ_m) / (wf · ρ_m + (1 − wf) · ρ_f)
    Returns: {"vf": float, "wf": float, "rho_f": float, "rho_m": float}
    """
    rho_f, rho_m = _get_densities(fiber_id, polymer_id)
    wf = mass_fraction
    vf = (wf * rho_m) / (wf * rho_m + (1.0 - wf) * rho_f)
    return {"vf": vf, "wf": wf, "rho_f": rho_f, "rho_m": rho_m}


def convert_volume_to_mass_fraction(
    fiber_id: int,
    polymer_id: int,
    volume_fraction: float,
) -> dict:
    """Convert fiber volume fraction (Vf) → mass fraction (wf).

    Formula: wf = (Vf · ρ_f) / (Vf · ρ_f + (1 − Vf) · ρ_m)
    Returns: {"wf": float, "vf": float, "rho_f": float, "rho_m": float}
    """
    rho_f, rho_m = _get_densities(fiber_id, polymer_id)
    vf = volume_fraction
    wf = (vf * rho_f) / (vf * rho_f + (1.0 - vf) * rho_m)
    return {"wf": wf, "vf": vf, "rho_f": rho_f, "rho_m": rho_m}


def get_model_inputs(
    fiber_id: int,
    polymer_id: int,
    use_inferred: bool = False,
) -> dict[str, float]:
    """Merge fiber datasheet + polymer datasheet, optionally overlaying inferred values.

    Returns model-unit dict ready to fill into any model's input vector.
    """
    inputs: dict[str, float] = {}
    inputs.update(get_fiber_inputs(fiber_id))
    inputs.update(get_polymer_inputs(polymer_id))

    if use_inferred:
        inputs.update(get_inferred_inputs(fiber_id, polymer_id))

    return inputs
