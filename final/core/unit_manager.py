"""unit_manager.py — singleton unit-conversion helper for DiffMicromechanics.

All three surrogates (elastic, thermoelastic, thermal) share the same
internal model units (MPa, 1/K, W/m·K, kg/m³).  This module maps every
field name to a physical quantity, looks up the display factor for the
currently active unit system, and provides bidirectional conversion.

Usage
-----
    from core.unit_manager import UM

    UM.set_system("GPa · µ/K · W/m·K")

    display_val = UM.to_display("E1", model_val)   # MPa  → GPa
    model_val   = UM.from_display("E1", user_val)  # GPa  → MPa
    label       = UM.unit_label("E1")              # "GPa"

    # Query a non-current system (useful for convert-on-switch logic)
    old_factor  = UM.get_factor("E1", old_system_name)

    # Convert directly between two named systems without changing current
    new_val = UM.convert_between("E1", value, from_system, to_system)

Adding a custom unit system
---------------------------
Edit  data/units.json  → add an entry to "unit_systems" that maps each
quantity key ("modulus", "cte", …) to one of the named system keys
already listed in "quantities"."modulus"."systems", etc.
Then restart the GUI — the new name will appear in the dropdown.
"""
from __future__ import annotations

import json
import os
from typing import Callable


class UnitManager:
    """Loads units.json and provides bidirectional unit conversion."""

    def __init__(self, config_path: str) -> None:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        # strip comment keys
        self._quantities: dict = {
            k: v for k, v in cfg["quantities"].items()
            if not k.startswith("_")
        }
        self._field_map:  dict[str, str] = cfg["field_quantity_map"]
        self._systems:    dict           = cfg["unit_systems"]
        self._current:    str            = next(iter(self._systems))
        self._callbacks:  list[Callable] = []

    #  public read-only properties 

    @property
    def current_system(self) -> str:
        return self._current

    @property
    def available_systems(self) -> list[str]:
        return list(self._systems.keys())

    # system selection 

    def set_system(self, name: str) -> None:
        """Change the active unit system and notify all registered callbacks."""
        if name not in self._systems:
            raise ValueError(f"Unknown unit system: '{name}'")
        self._current = name
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass

    def register_callback(self, cb: Callable) -> None:
        """Register a zero-argument callable to be invoked on every system change."""
        self._callbacks.append(cb)

    #  core factor lookup 

    def get_factor(self, field: str, system: str | None = None) -> float:
        """Return  display_value / model_value  for *field* in *system*.

        Uses the current system when *system* is None.
        Unknown fields are treated as dimensionless (factor = 1.0).
        """
        if system is None:
            system = self._current
        qty     = self._field_map.get(field, "dimensionless")
        sys_key = self._systems[system].get(qty, "default")
        return float(self._quantities[qty]["systems"][sys_key]["factor"])

    def unit_label(self, field: str, system: str | None = None) -> str:
        """Return the unit string for *field* in *system* (default: current)."""
        if system is None:
            system = self._current
        qty     = self._field_map.get(field, "dimensionless")
        sys_key = self._systems[system].get(qty, "default")
        return self._quantities[qty]["systems"][sys_key]["unit"]

    #  bidirectional conversion 

    def to_display(self, field: str, model_value: float) -> float:
        """Convert a model-unit value to the current display unit."""
        return model_value * self.get_factor(field)

    def from_display(self, field: str, display_value: float) -> float:
        """Convert a display-unit value back to model units."""
        factor = self.get_factor(field)
        return display_value / factor if factor != 0.0 else display_value

    def convert_between(self, field: str, value: float,
                        from_system: str, to_system: str) -> float:
        """Convert *value* for *field* from one named system to another.

        Does NOT change the active system.
        Useful for converting existing GUI values when the user switches systems.
        """
        f_from = self.get_factor(field, from_system)
        f_to   = self.get_factor(field, to_system)
        if f_from == 0.0:
            return value
        return value * f_to / f_from


#  module-level singleton 
_HERE = os.path.dirname(os.path.abspath(__file__))
# data/units.json lives in final/, one level up from final/core/
UM = UnitManager(os.path.join(os.path.dirname(_HERE), "data", "units.json"))
