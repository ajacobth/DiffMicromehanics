"""core/services — service layer for DiffMicromechanics.

All business logic lives here.  GUI files, CLI scripts, and agentic tool loops
import from this package — never from db.db or core.* directly.
"""

from core.services.service_material import (
    list_fibers,
    list_polymers,
    list_printers,
    list_cards,
    get_fiber_inputs,
    get_polymer_inputs,
    get_inferred_inputs,
    has_inferred_data,
    get_model_inputs,
    add_fiber,
    add_polymer,
    add_printer,
)

from core.services.service_forward import (
    get_model,
    run_forward,
    get_input_fields,
    get_output_fields,
)

from core.services.service_inverse import (
    run_inverse,
    validate_orientation_tensor,
    assemble_full_inputs,
    InverseResult,
)

from core.services.service_thermal import (
    load_thermal_data,
    vf_to_wf,
    run_thermal_inverse,
    ThermalResult,
)

from core.services.service_fim import run_fim

from core.services.service_cards import (
    load_card,
    load_card_inputs,
    save_inverse_result,
    save_forward_result,
    save_thermal_result,
    save_transfer_result,
)

from core.services.service_transfer import (
    resolve_constituent_props,
    build_forward_inputs,
    run_transfer,
    TransferResult,
)

__all__ = [
    # material
    "list_fibers", "list_polymers", "list_printers", "list_cards",
    "get_fiber_inputs", "get_polymer_inputs",
    "get_inferred_inputs", "has_inferred_data", "get_model_inputs",
    "add_fiber", "add_polymer", "add_printer",
    # forward
    "get_model", "run_forward", "get_input_fields", "get_output_fields",
    # inverse
    "run_inverse", "validate_orientation_tensor", "assemble_full_inputs", "InverseResult",
    # thermal
    "load_thermal_data", "vf_to_wf", "run_thermal_inverse", "ThermalResult",
    # fim
    "run_fim",
    # cards
    "load_card", "load_card_inputs",
    "save_inverse_result", "save_forward_result",
    "save_thermal_result", "save_transfer_result",
    # transfer
    "resolve_constituent_props", "build_forward_inputs", "run_transfer", "TransferResult",
]
