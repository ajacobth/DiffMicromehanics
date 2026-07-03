# Batch Pipeline Plan — Run All 3 Inverse Stages from a Single File

## Problem

Currently, running all 3 inverse stages (elastic → thermoelastic → thermal) requires
the user to interact with the agent step by step: provide measurements, wait for Stage 1
results, confirm, then provide Stage 2 inputs, etc. For users who already have all their
experimental data ready, this is unnecessary friction.

## Goal

Allow a user to provide a single Excel file containing all measurements and have the
agent automatically run Stage 1 → Stage 2 → Stage 3 in sequence, save results to a
card after each stage, and return a combined report — **without any changes to the
existing step-by-step interactive path**.

---

## Design Decision: New Tool, No Graph Changes

The sequencing logic lives inside a new Python tool (`run_full_pipeline`), not in the
graph. The LLM calls one tool; the checker sees one result. The existing
`agent → tool_executor → checker → agent` graph is unchanged.

**Why not rewire the graph?**
Routing each intermediate stage through the checker node in batch mode would require a
parallel subgraph, conditional edges, and state flags — significant complexity with no
real benefit. The tool can perform its own internal quality checks between stages and
abort early with a clear error if any stage fails.

**Tradeoff:** In the interactive flow the LLM decides whether to include CTE33 based on
Stage 1 quality. In batch mode this decision must be hardcoded: include CTE33 if it is
present in the file, exclude it otherwise. This is acceptable — batch users are opting
into automation.

---

## Design Principle: File = Lab Data Only

The Excel file contains **only experimental measurements** — the numbers the user
generated in the lab. Material identity (fiber, polymer, printer) and any known
microstructure parameters (fiber orientation, aspect ratio) are provided conversationally
to the agent, not in the file.

**Why:** Lab data is naturally tabular and file-based. Material context is a few words
the user already knows and tells the agent. Mixing them forces the user to learn a
key-value file format for information they could just say. It also makes the file
reusable — the same measurement file can be re-run against a different printer without
editing it.

---

## Agent-Assisted Material Population

Before calling `run_full_pipeline`, the agent resolves the material context from the
conversation:

1. **User states the material system** — e.g. "I want to run the full pipeline for T300
   carbon fiber with PESU on my Markforged X7"
2. **Agent confirms materials are in the DB** — calls `list_materials()` or
   `get_material_details()` to verify fiber and polymer exist
3. **If not in DB** — agent calls `add_fiber()` / `add_polymer()` first, guiding the user
   through the datasheet properties needed
4. **Agent asks about known microstructure** — "Do you have a known fiber mass fraction
   or orientation tensor from CT or process parameters? If not, the solver will infer them."
5. **User provides file path** — agent calls `run_full_pipeline(file_path, fiber_name,
   polymer_name, printer_name, [optional microstructure])`

This mirrors the existing step-by-step flow — the agent always establishes material
context before running any solver.

---

## Input File Format

Excel workbook (`.xlsx`) with two sheets — **measurements only, no material identity**.

### Sheet 1: `measurements`

Key-value layout (column A = field name, column B = value).
Only include the measurements you actually have — omit or leave blank to exclude.

**Elastic (Stage 1):**

| Field | Units | Notes |
|---|---|---|
| `E1_MPa` | MPa | Omit to exclude |
| `E2_MPa` | MPa | |
| `E3_MPa` | MPa | |
| `G12_MPa` | MPa | |
| `G13_MPa` | MPa | |
| `G23_MPa` | MPa | |
| `nu12` | — | |
| `nu13` | — | |
| `nu23` | — | |
| `E1_sigma_MPa` | MPa | 1-sigma uncertainty; use 0.0 if unknown |
| *(same pattern for other elastic sigmas)* | | |

**Thermoelastic (Stage 2):**

| Field | Units | Notes |
|---|---|---|
| `CTE11_per_K` | 1/K | Required for Stage 2. Convert ppm/K × 1e-6 |
| `CTE22_per_K` | 1/K | Required for Stage 2 |
| `CTE33_per_K` | 1/K | Optional — omit to exclude |
| `CTE11_sigma_per_K` | 1/K | |
| `CTE22_sigma_per_K` | 1/K | |
| `CTE33_sigma_per_K` | 1/K | |

### Sheet 2: `thermal`

Tabular layout. Required columns: `temperature_C`, `K11_WmK`.
Optional: `K22_WmK`, `K33_WmK`.

```
temperature_C  K11_WmK  K22_WmK
25             5.2      1.1
50             5.0      1.0
75             4.8      0.95
```

If the `thermal` sheet is absent or empty, Stage 3 is skipped and a note is included
in the report.

---

## Files to Create

### `core/services/service_pipeline.py` (new)

Responsible for:
1. Reading and validating the Excel file (`openpyxl`)
2. Extracting scalar inputs from Sheet 1 into a typed dict
3. Extracting thermal k vs T data from Sheet 2 into a temp CSV (reusing the existing
   thermal solver's CSV interface)
4. Calling the 3 solver functions in sequence — the same logic already inside each
   `run_*_inverse` tool, called directly (no LLM, no tool dispatch)
5. Saving to card after each successful stage via `service_cards`
6. Returning a structured result dict with per-stage status, inferred values, and
   fit errors

Key function signature:
```python
def run_full_pipeline(file_path: str) -> dict:
    """
    Returns:
    {
        "card_id": int,
        "card_name": str,
        "stage1": {"status": "pass"|"fail", "fit_error": float, "inferred": {...}},
        "stage2": {"status": "pass"|"fail"|"skipped", ...},
        "stage3": {"status": "pass"|"fail"|"skipped", ...},
        "errors": [str],   # one entry per failed stage
    }
    """
```

Abort rules:
- Stage 1 fail → abort; return with stage2/stage3 as "skipped"
- Stage 2 fail → abort stage3; still report stage1 results
- Stage 3 fail → report stages 1+2, note stage3 failure

### `agent/agent_tools.py` — add one tool

Material identity and any known microstructure parameters come from the conversation
(the agent extracts them); the file contains only measurements.

```python
@tool
def run_full_pipeline(
    file_path: str,
    fiber_name: str,
    polymer_name: str,
    printer_name: str,
    fiber_massfrac: float = -1.0,   # -1.0 = solver infers
    aspect_ratio: float = -1.0,     # -1.0 = solver infers
    a11: float = -1.0,              # -1.0 = solver infers
    a22: float = -1.0,              # -1.0 = solver infers
) -> str:
    """
    Run all 3 inverse stages (elastic → thermoelastic → thermal) from a measurements
    file, saving results to a material card automatically.

    Use this when the user provides a file path (.xlsx) containing experimental
    measurements. Do NOT use if the user is providing measurements directly in the
    conversation — use run_elastic_inverse instead.

    The file must have two sheets:
      - 'measurements': key-value table of elastic and CTE measurements (omit any
        you don't have). Do NOT include fiber/polymer/printer names here.
      - 'thermal': temperature_C and K11_WmK columns for Stage 3 (optional sheet)

    Material identity (fiber_name, polymer_name, printer_name) must be confirmed
    present in the database before calling this tool — use list_materials() or
    get_material_details() first, and add_fiber()/add_polymer() if needed.

    Microstructure params (fiber_massfrac, aspect_ratio, a11, a22) are optional —
    pass values only if the user has provided them from CT or process data.
    Pass -1.0 (default) to let the solver infer them.

    Results are saved to a card automatically.
    Returns a summary report of all 3 stages with inferred values and fit quality.
    """
```

### No other files change

- `agent/graph.py` — unchanged
- `agent/state.py` — unchanged
- All existing `run_*_inverse` tools — unchanged
- `config/quality_checklist.yaml` — may need a new `run_full_pipeline` entry for the
  checker node (optional; the tool report already includes per-stage fit errors)

---

## Template Excel File

Provide `data/pipeline_template.xlsx` so users don't have to build the file from
scratch. Contains only measurement fields — no material identity fields.
Pre-filled with example T300/PESU measurement values and placeholder rows for thermal data.
Include a comment row at the top of each sheet explaining the unit conventions.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| File not found | Return error immediately |
| Missing required field in Sheet 1 | List all missing fields, abort |
| Material name not in DB | Return name + suggestion to use `add_fiber`/`add_polymer` |
| Stage 1 solver fails (bad fit) | Report fit error, abort stages 2+3 |
| Stage 2 solver fails | Report, skip stage 3 |
| No `thermal` sheet | Skip stage 3, note in report |
| `openpyxl` not installed | Return install instruction |

---

## Checker Node Compatibility

The checker node regex-parses tool output for keywords (`fit_error`, `PASS`, `FAIL`,
inferred value ranges). The `run_full_pipeline` tool output should include the same
keywords so the existing checker rules fire correctly. Example output format:

```
=== BATCH PIPELINE REPORT ===

STAGE 1 — Elastic Inverse: PASS
  fit_error = 0.0023
  matrix_E = 3820 MPa,  matrix_poisson = 0.36
  a11 = 0.78,  a22 = 0.12,  fiber_massfrac = 0.44,  ar = 20.0

STAGE 2 — Thermoelastic Inverse: PASS
  fit_error = 0.0041
  f_cte1 = -0.5e-6 /K,  f_cte2 = 8.0e-6 /K,  m_cte = 52e-6 /K

STAGE 3 — Thermal Inverse: PASS
  fit_error = 0.0089
  k_f1 = 8.5 W/m·K,  k_f2 = 1.1 W/m·K
  p1 = 0.012,  p2 = 0.31

Card saved: "T300/PESU/X7"  (card_id = 4)
```

---

## Conversation Flow (Example)

```
User:   I want to run the full characterisation pipeline for T300 carbon fiber
        with PESU on my Markforged X7. Here's my data file: /data/t300_pesu.xlsx

Agent:  [calls list_materials() → confirms T300 and PESU are in the DB]
        Got it — T300 (Toray) and PESU Ultrason (BASF) are both in the library.
        Do you have a known fiber mass fraction or fiber orientation tensor from
        CT or process parameters, or should the solver infer them?

User:   I know the mass fraction is 0.44 but not the orientation.

Agent:  [calls run_full_pipeline(
            file_path="/data/t300_pesu.xlsx",
            fiber_name="Carbon Fiber T300",
            polymer_name="PESU Ultrason",
            printer_name="Markforged X7",
            fiber_massfrac=0.44
        )]
```

---

## Build Order

1. `core/services/service_pipeline.py` — solver logic + Excel parsing
2. `data/pipeline_template.xlsx` — measurements-only template with unit comments
3. Add `run_full_pipeline` tool to `agent/agent_tools.py`
4. (Optional) Add `run_full_pipeline` quality rules to `config/quality_checklist.yaml`
5. Update `AGENT_SETUP.md` section on batch usage

---

## Out of Scope

- GUI integration (batch upload button in Streamlit) — future work
- Partial reruns (e.g. re-run only Stage 3 from an existing card) — use the existing
  `run_thermal_inverse` tool for that
- Auto-detection of units — file must use the units specified in the template
