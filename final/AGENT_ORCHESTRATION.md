# Agent Orchestration — Design Document

This document describes the architecture for adding a conversational AI agent to
DiffMicromechanics. It covers the design philosophy, the three files that make up
the agent layer, the technology stack, example conversation flows, and the
long-term fine-tuning roadmap.

---

## Table of Contents

1. [What the Agent Does](#1-what-the-agent-does)
2. [Design Philosophy](#2-design-philosophy)
3. [Architecture Overview](#3-architecture-overview)
4. [File 1 — Vocabulary (`vocabulary.md`)](#4-file-1--vocabulary-vocabularymd)
5. [File 2 — Agent Tools (`agent_tools.py`)](#5-file-2--agent-tools-agent_toolspy)
6. [File 3 — System Prompt (`system_prompt.md`)](#6-file-3--system-prompt-system_promptmd)
7. [How the Three Files Work Together](#7-how-the-three-files-work-together)
8. [Technology Stack](#8-technology-stack)
9. [Example Conversation Flows](#9-example-conversation-flows)
10. [Fine-Tuning Roadmap](#10-fine-tuning-roadmap)
11. [Hardware Requirements](#11-hardware-requirements)
12. [LangGraph — State Tracking and Routing](#12-langgraph--state-tracking-and-routing)
13. [Session Persistence and Memory Restoration](#13-session-persistence-and-memory-restoration)
14. [Intent Routing — Mode-Based Tool Filtering](#14-intent-routing--mode-based-tool-filtering)

---

## 1. What the Agent Does

The agent is a conversational interface for material suppliers who need to
characterize fiber-reinforced composites. Instead of operating the GUI or writing
scripts, a user can ask in plain English:

> "Can you tell me the orientation values given E1 is 45 GPa and E2 is 12 GPa?"

> "I have K vs T data for my composite — can you find the fiber and matrix
> conductivities?"

> "We just got a new printer. Can you predict how our Carbon/PESU material will
> behave on it?"

The agent interprets the request, determines what is needed to answer it, asks
follow-up questions when the problem is under-specified, calls the appropriate
computation service, and explains the result.

Critically, the agent runs **entirely offline** — no internet connection is required
at runtime. This is a firm requirement for use by material suppliers with
proprietary data.

---

## 2. Design Philosophy

### Intelligence belongs in the model, not the tools

There are two ways to design an agentic system:

**Tool-smart approach**: The tools do the reasoning. The agent calls
`check_identifiability()` and the tool returns a pre-computed recommendation:
"Add G12 as a target." The model just relays this to the user.

**Model-smart approach**: The model does the reasoning. The model knows that
E1 + E2 is insufficient to identify a11, a22, fiber_massfrac, and ar
simultaneously — because this physics knowledge is encoded in the system prompt
and internalized through fine-tuning. The model asks the right follow-up
questions before calling any tool.

This codebase takes the **model-smart approach**. The rationale:

1. Tools that embed intelligence are hard to maintain. Every new material system
   or new identifiability insight requires code changes. A well-prompted model
   generalises.

2. Material suppliers ask unexpected questions. A tool-smart system can only
   handle the cases its tools were written for. A model-smart system can reason
   about novel combinations it has never explicitly seen.

3. The model-smart approach is the path to fine-tuning. Once the model handles
   things correctly from the system prompt, those conversations become training
   data. Eventually the knowledge moves from the prompt into the weights.

### Tools are thin wrappers, not orchestrators

`agent_tools.py` calls the service layer and returns results. It does not decide
what to call, when to call it, or what to tell the user. Those decisions belong
to the model. The tools are execution only.

### Offline first

The agent must run without any network calls at inference time. This means:

- The language model is served locally via Ollama
- No cloud APIs are called during a conversation
- The model, its weights, and all tools run on the user's hardware

The only network activity allowed is the one-time model download during initial
setup, which can be done in a controlled environment before deployment to the
supplier.

---

## 3. Architecture Overview

```
User (plain English)
        │
        ▼
┌───────────────────┐
│  Language Model   │  ← system_prompt.md  (physics knowledge)
│  (local, Ollama)  │  ← vocabulary.md   (field name mapping)
└───────────────────┘
        │  tool calls
        ▼
┌───────────────────┐
│  agent_tools.py   │  ← thin wrappers, no intelligence
└───────────────────┘
        │  function calls
        ▼
┌───────────────────┐
│  Service Layer    │  ← service_inverse, service_forward,
│  (core/services/) │     service_thermal, service_transfer,
└───────────────────┘     service_cards, service_material
        │
        ▼
┌───────────────────┐
│  Database + Models│  ← data/micromechanics.db, models/
└───────────────────┘
```

Three files form the agent layer:

| File | Job | Changes when? |
|---|---|---|
| `vocabulary.md` | Maps user language → field names and units | When new materials or properties are added |
| `agent_tools.py` | Exposes service layer to LangChain | When service layer API changes |
| `system_prompt.md` | Encodes physics knowledge and reasoning rules | When understanding of the domain deepens, or fine-tuning makes parts of it redundant |

The service layer beneath is unchanged — the agent is a new consumer of an
already-clean API.

---

## 4. File 1 — Vocabulary (`vocabulary.md`)

### Purpose

The vocabulary file is the bridge between how users talk about materials and
how the models represent them. Users say "fiber alignment" — the model field is
`a11`. Users say "stiffness in the print direction" — the model field is `E1`
in MPa. Without an explicit vocabulary, a small model will hallucinate field
names or confuse units.

### What it contains

**Orientation and microstructure:**

```json
{
  "orientation": {
    "synonyms": ["fiber alignment", "orientation tensor", "fiber orientation",
                 "anisotropy", "orientation state"],
    "fields": ["a11", "a22", "a12", "a13", "a23"],
    "description": "Second-order orientation tensor components. a11 is alignment
                    along the print direction (x). a22 is alignment in the
                    transverse direction (y). a33 = 1 - a11 - a22 is inferred
                    automatically. Off-diagonals a12, a13, a23 are shear
                    coupling terms, usually small.",
    "units": "dimensionless",
    "typical_range": {"a11": [0.3, 0.9], "a22": [0.05, 0.4]}
  },

  "aspect_ratio": {
    "synonyms": ["fiber shape", "fiber length", "l/d ratio", "slenderness"],
    "fields": ["ar"],
    "description": "Ratio of fiber length to diameter. Higher values mean
                    longer fibers. Short-fiber composites are typically 5-30.
                    Continuous fiber approaches infinity but is not handled
                    by this model.",
    "units": "dimensionless",
    "typical_range": [5, 100]
  },

  "fiber_content": {
    "synonyms": ["fiber mass fraction", "fiber loading", "reinforcement content",
                 "fiber weight fraction", "wf"],
    "fields": ["fiber_massfrac"],
    "description": "Mass of fiber divided by total composite mass. Not the
                    same as volume fraction.",
    "units": "dimensionless (0.01 to 0.60)"
  }
}
```

**Elastic properties:**

```json
{
  "E1": {
    "synonyms": ["axial modulus", "modulus in print direction", "longitudinal
                  stiffness", "E along x"],
    "description": "Young's modulus along the fiber/print direction.",
    "units": "MPa",
    "conversion": {"GPa": 1000}
  },
  "E2": {
    "synonyms": ["transverse modulus", "in-plane transverse stiffness"],
    "description": "Young's modulus transverse to the print direction,
                    in the build plane.",
    "units": "MPa"
  },
  "G12": {
    "synonyms": ["shear modulus", "in-plane shear stiffness"],
    "units": "MPa"
  },
  "nu12": {
    "synonyms": ["Poisson ratio", "in-plane Poisson", "lateral contraction"],
    "units": "dimensionless"
  }
}
```

**Thermal properties:**

```json
{
  "CTE11": {
    "synonyms": ["thermal expansion in print direction", "alpha 1",
                 "axial CTE", "longitudinal thermal expansion"],
    "units": "1/K",
    "conversion": {"ppm/K": 1e-6}
  },
  "K11": {
    "synonyms": ["thermal conductivity along print direction", "axial
                  conductivity", "k parallel"],
    "units": "W/m·K"
  }
}
```

**Constituent properties:**

```json
{
  "matrix_modulus": {
    "synonyms": ["matrix stiffness", "polymer modulus", "resin modulus",
                 "matrix Young's modulus"],
    "units": "MPa",
    "note": "This is an inferred value in Stage 1. The datasheet value is
             the starting point but the actual in-situ value often differs."
  },
  "f_cte1": {
    "synonyms": ["fiber axial CTE", "fiber longitudinal thermal expansion",
                 "alpha fiber 1"],
    "units": "1/K",
    "note": "Inferred in Stage 2. Fiber CTE values from datasheets are
             unreliable — always infer from composite measurements."
  }
}
```

### Format notes

- JSON so it can be loaded by both the agent and any validation tooling
- Synonyms list covers what real users actually say, including informal terms
- Units and conversions are explicit — users often supply GPa when the model
  expects MPa
- Typical ranges help the model flag implausible values before running
  expensive solves

### How the model uses it

The vocabulary file is either:
- Injected into the system prompt (at the cost of prompt length), or
- Loaded as a retrieval tool that the model queries when it needs to map a term

For small models (≤13B), injection is more reliable. For larger models or
after fine-tuning, the vocabulary knowledge lives in the weights and the file
is only needed for validation.

---

## 5. File 2 — Agent Tools (`agent_tools.py`)

### Purpose

`agent_tools.py` exposes the service layer to LangChain as a set of callable
tools. The model sees tool names, descriptions, and argument schemas. It
decides what to call — the file just executes the call and returns a
human-readable result string.

### Design rules for this file

1. **No intelligence here.** No recommendations, no identifiability checks,
   no pre-computed guidance. The model reasons — the tools execute.

2. **Flat argument signatures.** Small models hallucinate nested dict
   structure. Individual named floats are far more reliable than
   `targets: dict[str, float]`.

3. **String returns.** LangChain feeds tool output back into the model as
   text. A formatted string is more readable to the model than a JSON blob.

4. **Errors as strings, not exceptions.** Unhandled exceptions terminate the
   agent loop. Errors should be caught and returned as descriptive strings so
   the model can reason about what went wrong and try a different approach.

5. **Units in return strings.** Always annotate units in tool output so the
   model can relay them correctly to the user.

### Function signatures

**Discovery tools** — always the first calls an agent makes:

```python
@tool
def list_materials() -> str:
    """List all available fibers, polymers, and printers in the database.
    Call this first to find the correct IDs before running any computation."""

@tool
def list_cards() -> str:
    """List all material cards (fiber + polymer + printer combinations)
    and their characterization status (which stages are complete)."""

@tool
def get_card_status(card_id: int) -> str:
    """Get the full characterization status of a material card: which stages
    are complete, what constituent properties have been inferred, and what
    microstructure values are stored."""
```

**Stage execution tools:**

```python
@tool
def run_elastic_inverse(
    fiber_id: int,
    polymer_id: int,
    printer_id: int,
    E1_MPa: float,
    E2_MPa: float,
    G12_MPa: float,
    nu12: float,
    E1_sigma_MPa: float = 0.0,
    E2_sigma_MPa: float = 0.0,
    G12_sigma_MPa: float = 0.0,
    nu12_sigma: float = 0.0,
    card_id: int = -1,
    card_name: str = "",
) -> str:
    """Run Stage 1 elastic inverse estimation. Infers matrix modulus, matrix
    Poisson ratio, and full microstructure (orientation tensor, fiber mass
    fraction, aspect ratio) from measured composite elastic properties.
    Sigma values represent measurement uncertainty (1 standard deviation).
    Set card_id=-1 to create a new card, or provide an existing card_id to
    update it. Returns inferred values, fit error, and card_id."""

@tool
def run_thermoelastic_inverse(
    card_id: int,
    CTE11_per_K: float,
    CTE22_per_K: float,
    CTE11_sigma_per_K: float = 0.0,
    CTE22_sigma_per_K: float = 0.0,
) -> str:
    """Run Stage 2 thermoelastic inverse estimation. Requires Stage 1
    (elastic inverse) to be complete on this card. Infers fiber axial CTE,
    fiber transverse CTE, and matrix CTE from measured composite thermal
    expansion coefficients. Microstructure and matrix modulus from Stage 1
    are used as fixed inputs."""

@tool
def run_thermal_inverse(
    card_id: int,
    csv_path: str,
    n_restarts: int = 5,
) -> str:
    """Run Stage 3 thermal inverse estimation. Requires Stage 1 to be
    complete. Loads temperature-dependent conductivity data from a CSV file
    (columns: temperature, K11, K22, K33 — temperature in Celsius,
    conductivity in W/m·K) and infers fiber longitudinal conductivity,
    fiber transverse conductivity, and the polymer conductivity parametric
    model (p1, p2 where K_m(T) = p1*sqrt(T/T_ref) + p2)."""

@tool
def run_transfer(
    source_card_id: int,
    target_printer_id: int,
    E1_MPa: float,
    E2_MPa: float,
    G12_MPa: float,
    nu12: float,
    E1_sigma_MPa: float = 0.0,
    E2_sigma_MPa: float = 0.0,
    G12_sigma_MPa: float = 0.0,
    nu12_sigma: float = 0.0,
) -> str:
    """Transfer material properties from a characterized source card to a
    new printer. Uses the source card's constituent properties (matrix
    modulus, CTEs, thermal conductivities) as fixed inputs and infers the
    new printer's microstructure from measured composite elastic properties.
    Then predicts all elastic, thermoelastic, and thermal composite
    properties for the new printer. Source card must have Stage 1 complete."""
```

**Forward prediction:**

```python
@tool
def predict_properties(
    card_id: int,
    a11: float = -1.0,
    a22: float = -1.0,
    fiber_massfrac: float = -1.0,
    ar: float = -1.0,
) -> str:
    """Run forward prediction for all three models (elastic, thermoelastic,
    thermal) using a material card's stored constituent properties. If
    microstructure values (a11, a22, fiber_massfrac, ar) are provided
    (positive values), they override the stored microstructure. Otherwise
    uses the most recent stored microstructure snapshot. Returns predicted
    E1, E2, G12, nu12, CTE11, CTE22, K11, K22, K33 and all other outputs."""
```

**Introspection tools** — model calls these when unsure about field names:

```python
@tool
def get_model_inputs(model_name: str) -> str:
    """Return the complete list of input fields required by a model.
    model_name must be 'elastic', 'thermoelastic', or 'thermal'.
    Use this to verify which fields a model needs before running."""

@tool
def get_model_outputs(model_name: str) -> str:
    """Return the complete list of output fields produced by a model.
    model_name must be 'elastic', 'thermoelastic', or 'thermal'."""
```

### What these signatures deliberately omit

- **Solver configuration** (method, tolerance, iterations) — defaulted
  internally. Advanced users can be directed to the GUI or CLI for fine control.
- **Free/fixed field selection** — hardcoded per stage based on physics.
  The agent does not need to know which fields are free in Stage 1.
- **Bounds** — defaulted internally to physically reasonable ranges.
- **Field aliases** (`fiber_massfrac` vs `w_f`, `ar` vs `ar_f`) — resolved
  internally. The agent always uses canonical names.

### Return string format

Each tool returns a structured, labelled string. Example from
`run_elastic_inverse`:

```
ELASTIC INVERSE — COMPLETE
card_id: 3
fit_error: 0.0023 (dimensionless, lower is better)

Inferred microstructure:
  a11 = 0.724  (fiber alignment along print direction)
  a22 = 0.141  (transverse alignment)
  a12 = 0.018, a13 = 0.009, a23 = 0.003
  fiber_massfrac = 0.348  (34.8% fiber by mass)
  aspect_ratio = 18.5

Inferred constituent properties:
  matrix_modulus = 3210 MPa
  matrix_poisson = 0.382

Model predictions at this microstructure:
  E1 = 44,820 MPa  (target: 45,000 MPa)
  E2 = 12,140 MPa  (target: 12,000 MPa)
  G12 = 4,480 MPa  (target: 4,500 MPa)
  nu12 = 0.312     (target: 0.310)
```

This format is readable by both the model (for chaining reasoning) and the
user (when the model relays results).

---

## 6. File 3 — System Prompt (`system_prompt.md`)

### Purpose

The system prompt is the model's physics knowledge base. It teaches the model:

- What composite micromechanics is and what this application does
- The three-layer material model (constituent / microstructure / composite)
- Which measurements identify which unknowns — identifiability
- The four-stage characterization workflow and why each stage exists
- How to handle underdetermined problems conversationally
- When to proceed vs when to ask for more information

### Structure

The system prompt is divided into six sections. The full content is written
separately in `system_prompt.md` — this document describes what each section
must contain.

---

**Section 1: Role and context**

Establishes who the model is and what it can do. Must be brief — 3-5 sentences.
States that it operates offline, that it has access to computation tools, and
that its primary job is to help characterize composite materials.

---

**Section 2: The three-layer material model**

Explains the fundamental physics separation that underpins the entire system:

- **Constituent properties** are intrinsic to the fiber and polymer materials
  and do not depend on the printer. Matrix modulus, fiber CTE, polymer thermal
  conductivity — these are the same regardless of which machine made the part.

- **Microstructure** is printer-specific. The orientation tensor, fiber mass
  fraction, and aspect ratio are determined by the printing process. The same
  material on a different printer will have different microstructure.

- **Composite properties** are derived from both. E1, CTE11, K11 — these
  depend on both what the materials are and how they are arranged.

This three-layer model is the physical justification for:
- Why characterization is staged (infer microstructure first from elastic,
  then use it as fixed when inferring CTE)
- Why constituent properties are printer-independent and can be transferred
- Why only microstructure needs to be re-identified on a new printer

---

**Section 3: The four-stage characterization workflow**

Explains each stage, what it infers, what it requires as input, and why the
ordering is mandatory:

**Stage 1 — Elastic inverse:**
Free variables: matrix modulus, matrix Poisson ratio, orientation tensor
(a11, a22, a12, a13, a23), fiber mass fraction, aspect ratio.
Required measurements: E1, E2, G12, nu12 (composite elastic properties).
Why this order: elastic properties are most sensitive to microstructure.
Oriention tensor is best constrained by the full set of elastic moduli.

**Stage 2 — Thermoelastic inverse:**
Free variables: fiber axial CTE (f_cte1), fiber transverse CTE (f_cte2),
matrix CTE (m_cte).
Fixed inputs: everything from Stage 1 (microstructure + matrix modulus).
Required measurements: CTE11, CTE22.
Why this order: CTE inference is only well-conditioned when microstructure is
already known. Trying to infer CTE and microstructure simultaneously is
severely underdetermined.
Note: CTE values are stored globally — they apply to any printer using the
same fiber and polymer, because CTE is a constituent property.

**Stage 3 — Thermal inverse:**
Free variables: fiber longitudinal conductivity (l2), fiber anisotropy ratio
(t = l2/k_f2), polymer conductivity parametric model (p1, p2).
Fixed inputs: microstructure from Stage 1.
Input data: K vs T curve from CSV (not target values — a full curve).
Note: The polymer conductivity model is K_m(T) = p1 * sqrt(T / T_ref) + p2,
where T_ref = 1°C. This parametric form captures the temperature dependence
of polymer conductivity. The inferred p1 and p2 are stored globally.

**Stage 4 — Transfer to new printer:**
Uses constituent properties from a characterized card (Stages 1-3).
Holds constituent properties fixed.
Infers new microstructure from measured elastic properties on the new printer.
Then predicts all composite properties for the new printer.
Requires: source card with at least Stage 1 complete.

---

**Section 4: Identifiability — the core reasoning skill**

This is the most important section. The model must understand which
combinations of measurements can identify which unknowns.

**The fundamental rule:**
The number of independent measurements must be at least equal to the number of
free variables for the problem to be well-posed. But this is necessary, not
sufficient — the measurements must also be sensitive to the unknowns they are
trying to identify.

**Known identifiability rules for the elastic model:**

| Free variables | Minimum measurements needed | Notes |
|---|---|---|
| a11, a22 only | E1, E2 | Marginally determined. G12 improves confidence. |
| a11, a22, fiber_massfrac | E1, E2, G12 | Well-determined. |
| a11, a22, fiber_massfrac, ar | E1, E2, G12, nu12 | Aspect ratio is weakly identified. |
| Full microstructure + matrix_modulus + matrix_poisson | E1, E2, G12, nu12 | Standard Stage 1. Well-determined. |
| Any microstructure + CTE simultaneously | Cannot be done reliably | Always run Stage 1 before Stage 2. |

**Known identifiability rules for thermal:**

| Free variables | Minimum data needed |
|---|---|
| p1, p2, l2, t | K11 vs T curve over at least 50°C range |
| p1, p2 only (fix fiber k) | K11 vs T at 3+ temperatures |

**How to handle underdetermined requests:**

When a user asks to infer more unknowns than their measurements can support,
the model must:
1. Explain clearly why the problem is underdetermined (in plain English, not
   math)
2. Offer options: add more measurements, fix some variables to datasheet
   values, or reduce the scope of what is being inferred
3. Never run the solver on an underdetermined problem without warning the user

Example: "I want to find the orientation from E1 alone"
Correct response: "E1 alone is not sufficient to uniquely identify all
orientation components and fiber content — the problem has 7 unknowns but
only 1 constraint. If you also have E2, G12, and nu12, I can run a reliable
inverse. Alternatively, if you can fix the aspect ratio to its datasheet
value and assume a planar orientation (a13 = a23 = 0), E1 + E2 may be
sufficient for a11, a22, and fiber_massfrac."

---

**Section 5: Unit handling**

The model must always handle unit conversion before calling any tool. All
tools expect model units (MPa, 1/K, W/m·K). Users commonly supply:

- Moduli in GPa → multiply by 1000 to get MPa
- CTE in ppm/K → multiply by 1e-6 to get 1/K
- Temperature in Fahrenheit → convert to Celsius before passing to tools

The model should confirm units with the user when there is ambiguity, and
always state units when reporting results.

---

**Section 6: Conversation guidelines**

Practical rules for how the model should conduct conversations:

- Ask for fiber, polymer, and printer identity early — most tool calls need
  material IDs.
- Call `list_materials()` at the start of any new characterization discussion
  to confirm what is in the database.
- Call `get_card_status()` before recommending what stage to run next — the
  user may have already done part of the characterization.
- Never guess at field values. If a required input is unknown, ask.
- After a successful solve, always report the fit error and explain what it
  means: fit_error < 0.01 is typically acceptable, > 0.1 suggests the
  measurements may be inconsistent with the model.
- If a solve fails or returns a poor fit, suggest possible causes: measurement
  error, wrong material assignment, or the problem being genuinely
  underdetermined.
- After reporting a successful solve result, ask the user if they want to save
  it to the card before proceeding. Results that are not saved are lost if the
  session ends. Do not save automatically — always ask first.
- If the user explicitly says to save (e.g. "save it", "yes save", "save and
  continue"), call `save_to_card` immediately without asking for confirmation
  again and move on.
- Make clear to the user that unsaved results will be lost if they close the
  session — this is the same contract as the GUI.

---

## 7. How the Three Files Work Together

A complete example showing all three files in action:

**User:** "Can you find the fiber orientation for my carbon/PESU composite?
I measured E1 = 40 GPa, E2 = 8 GPa."

**Model (using vocabulary.md):**
Maps "fiber orientation" → a11, a22, a12, a13, a23.
Converts 40 GPa → 40,000 MPa, 8 GPa → 8,000 MPa.

**Model (using system_prompt.md, Section 4):**
Counts unknowns: a11, a22, a12, a13, a23, fiber_massfrac, ar = 7 unknowns.
Counts measurements: E1, E2 = 2.
Identifies: underdetermined. E1 + E2 alone cannot identify 7 unknowns.

**Model responds:**
"I can work towards finding the fiber orientation, but E1 and E2 alone are not
sufficient to uniquely determine all the orientation tensor components plus
fiber content and aspect ratio — that is 7 unknowns with only 2 measurements.

To run a reliable elastic inverse I would need at minimum:
- G12 (in-plane shear modulus)
- nu12 (in-plane Poisson ratio)

Do you have these measurements available? If not, I can fix the aspect ratio
to a typical value and assume symmetric orientation (a12=a13=a23=0), which
reduces the problem to 3 unknowns (a11, fiber_massfrac, matrix_modulus) —
but this assumes the out-of-plane orientation is negligible."

**User:** "I also have G12 = 3.8 GPa and nu12 = 0.34"

**Model:**
Now has 4 measurements for 7 unknowns (still needs matrix_modulus and
matrix_poisson as additional free variables). Recognises this is the standard
Stage 1 setup. Checks: is there an existing card for Carbon/PESU?

**Model calls:** `list_materials()` → confirms Carbon Fiber T300, PESU Ultrason
are in database. Calls `list_cards()` → no existing card for this material/printer.

**Model calls:**
```
run_elastic_inverse(
  fiber_id=1, polymer_id=1, printer_id=?,
  E1_MPa=40000, E2_MPa=8000, G12_MPa=3800, nu12=0.34,
  card_name="Carbon T300 / PESU — Stage 1"
)
```

**Tool executes:** calls service_inverse → service_cards → returns result string.

**Model responds:** Reports inferred orientation values with units and fit error.
Offers to run Stage 2 if the user has CTE measurements.

---

## 8. Technology Stack

### Language model serving: Ollama

Ollama is the recommended local model server. It:
- Runs entirely offline after initial model download
- Supports tool calling for capable models
- Integrates directly with LangChain via `ChatOllama`
- Manages model quantization and memory automatically

```bash
# One-time setup (can be done before deploying to supplier)
# Use 32B on 32 GB machines; switch to 70B only on 64 GB+ hardware
ollama pull llama3.3:32b-instruct-q4_K_M

# Runtime (no network needed)
ollama serve
```

### Agent framework: LangChain + LangGraph

LangChain provides:
- `@tool` decorator that generates JSON schemas from Python type hints and
  docstrings — these schemas are what the model sees when deciding what to call
- `ChatOllama` class for communicating with the local Ollama server
- `ToolNode` for executing tool calls returned by the model

LangGraph (built on top of LangChain) is used specifically for:
- Persistent `AgentState` across conversation turns (completed stages, locked
  inputs, current card)
- In-memory `MemorySaver` checkpointing within a session. On resume, state is
  reconstructed from existing DB tables — no checkpoint blobs written to disk
- The `stage_updater` node that updates state after solver tools complete

LangGraph is deliberately used only for these two jobs. All reasoning and
routing remains in the model. See Section 12 for the full graph design.

The entry point is a small `scripts/run_agent.py` that:
1. Loads the system prompt from `system_prompt.md`
2. Loads vocabulary from `vocabulary.md` (injected into system prompt or
   available as a tool)
3. Registers all tools from `agent_tools.py`
4. Creates a `ChatOllama` model instance
5. Compiles the LangGraph with a SQLite checkpointer
6. Starts a conversation loop

### Model selection

The minimum viable model for this use case depends on what is expected of it:

| Model | RAM needed (Q4) | Capability |
|---|---|---|
| Llama 3.1 8B Q4 | ~5 GB | Basic tool calling. Reliable for simple, single-stage requests. Will struggle with identifiability reasoning on edge cases. |
| Phi-4 14B Q4 | ~10 GB | Better reasoning per parameter. More reliable on multi-step conversations. |
| Llama 3.3 32B Q4 | ~20 GB | Good reasoning. Handles multi-stage workflows and most identifiability cases reliably. Best option for 32 GB machines. |
| Qwen2.5 32B Q4 | ~20 GB | Strong on technical/scientific tasks. Comparable to Llama 3.3 32B; worth benchmarking both. |
| Llama 3.3 70B Q4 | ~40 GB | Best reasoning before fine-tuning. Requires 48+ GB RAM — does not fit in 32 GB. |
| Qwen2.5 72B Q4 | ~40 GB | Best domain performance before fine-tuning. Also requires 48+ GB RAM. |

**For 32 GB machines:** Llama 3.3 32B Q4 or Qwen2.5 32B Q4. Both fit
comfortably (~20 GB), leaving headroom for the OS and application. The 70B
models require ~40 GB and will not run reliably on 32 GB hardware — use 32B
as the development baseline, switch to 70B only on a larger workstation.

**After fine-tuning** (see Section 10), the same quality can be achieved with an
8B model running on a laptop.

---

## 9. Example Conversation Flows

### Flow A — Standard four-stage characterization

```
User:   I want to characterize my E-Glass / Epoxy composite on my Markforged X7.

Agent:  [calls list_materials() — confirms E-Glass and Epoxy 3501-6 in DB]
        [calls list_cards() — no existing card for Markforged X7]
        Great. To characterize your composite I will need measurements at each
        stage. Let's start with Stage 1 (elastic properties). Do you have E1,
        E2, G12, and nu12 measurements for this composite?

User:   Yes. E1=38 GPa, E2=11 GPa, G12=4.1 GPa, nu12=0.28.

Agent:  [validates units — converts to MPa]
        [calls run_elastic_inverse(...)]
        Stage 1 complete. Here is what was inferred: [results]. Fit error is
        0.0031 — good fit. The matrix modulus came out at 4,100 MPa, which is
        within the expected range for Epoxy 3501-6 in-situ.

        Do you have CTE measurements (CTE11 and CTE22) to proceed to Stage 2?

User:   CTE11 = 3.2 ppm/K, CTE22 = 28 ppm/K.

Agent:  [converts ppm/K → 1/K]
        [calls run_thermoelastic_inverse(card_id=..., ...)]
        Stage 2 complete. Inferred: [CTE results].

        Do you have thermal conductivity vs temperature data for Stage 3?

User:   Yes, I have a CSV at /data/eglass_epoxy_k.csv

Agent:  [calls run_thermal_inverse(card_id=..., csv_path="...")]
        Stage 3 complete. All four characterization stages done.
        Your material card now has full elastic, thermoelastic, and thermal
        characterization.
```

### Flow B — Transfer to new printer

```
User:   We just got an Anisoprint A4. Can you predict how our Carbon/PESU
        material will behave on it?

Agent:  [calls list_cards() — finds Carbon/PESU Markforged X7 card, Stages 1-3 complete]
        [calls get_card_status(card_id=3)]
        Found your Carbon T300 / PESU card from the Markforged X7 with full
        characterization. To predict properties on the Anisoprint A4, I need
        elastic measurements from a part printed on it. Do you have E1, E2,
        G12, and nu12 for a Carbon/PESU part from the Anisoprint?

User:   E1=35 GPa, E2=9.5 GPa, G12=3.6 GPa, nu12=0.31.

Agent:  [calls run_transfer(source_card_id=3, target_printer_id=2, ...)]
        Transfer complete. The Anisoprint A4 produces lower a11 (0.61 vs 0.72
        on the Markforged) and slightly lower fiber content. Here are the
        predicted properties: [full elastic, thermoelastic, thermal results].
        These use your previously inferred constituent properties (matrix
        modulus, CTEs, thermal conductivities) as fixed values — only the
        microstructure was re-identified.
```

### Flow C — Flexible user request with identifiability guidance

```
User:   Can you tell me the orientation values if E1 is 45 GPa?

Agent:  [system_prompt knowledge: E1 alone → 1 measurement, 7 unknowns]
        E1 alone is not enough to uniquely determine the full orientation
        tensor — there are 7 microstructure unknowns (a11, a22, a12, a13,
        a23, fiber mass fraction, aspect ratio) but only 1 measurement.

        The minimum set that gives a reliable result is E1, E2, G12, and nu12.

        If you only have E1, I can still give you a rough estimate by:
        (a) Fixing aspect ratio to a datasheet value
        (b) Assuming planar orientation (a12 = a13 = a23 = 0)
        (c) Fixing fiber mass fraction to a nominal value
        This reduces the problem to 2 unknowns (a11, a22) — marginally
        solvable from E1 + E2.

        What other measurements do you have?
```

---

## 10. Fine-Tuning Roadmap

The system prompt is a transitional scaffold. The long-term goal is to move
the physics knowledge from the prompt into the model weights through fine-tuning,
so the agent can run on a small model with a minimal prompt.

### Phase 1 — Bootstrap (current design)

Deploy the system prompt + vocabulary + tool file with a large local model
(70B). Log every conversation from day one. Each log entry records:
- User messages
- Model reasoning steps
- Tool calls and results
- Final responses

### Phase 2 — Dataset curation

After collecting 200-500 real conversations, curate a fine-tuning dataset:

- Keep conversations where the agent behaved correctly (right tool called,
  right identifiability guidance given, correct unit conversion)
- Discard conversations with errors or corrections
- Augment with synthetic conversations generated by a teacher model (Claude
  or GPT-4): feed the teacher the system prompt and ask it to generate
  variations of the real conversations covering edge cases

Target dataset size:
- Vocabulary + field mapping: 200-500 examples
- Stage workflow + tool calling: 300-500 examples
- Identifiability reasoning: 500-1000 examples (harder, needs more coverage)
- Total: approximately 1000-2000 high-quality conversations

### Phase 3 — Fine-tuning

Use QLoRA (Quantized Low-Rank Adaptation) to fine-tune a base model:

- **Base model:** Llama 3.1 8B or Qwen2.5 7B
- **Method:** QLoRA with 4-bit quantization — runs overnight on a single
  consumer GPU (RTX 4090, 24 GB VRAM)
- **Framework:** Unsloth (fastest QLoRA implementation) or Axolotl
- **Data format:** ShareGPT format (system, user, assistant, tool_call turns)

QLoRA preserves the base model's general capabilities while adding domain
knowledge. The LoRA adapter is a small file (50-200 MB) that can be distributed
separately from the base model weights.

### Phase 4 — Evaluation and iteration

Test the fine-tuned model against a held-out evaluation set derived from the
system prompt — every scenario the system prompt describes should be tested:

- Does it correctly map "orientation" to a11, a22?
- Does it refuse to run an underdetermined inverse without warning?
- Does it convert GPa to MPa before calling tools?
- Does it ask for G12 when the user only provides E1 + E2?

For every failure case, add a corrective example to the training data and
retune. This is an iterative process — expect 3-5 cycles.

### Phase 5 — Prompt reduction

Once the fine-tuned model passes evaluation, begin reducing the system prompt:

1. Remove the vocabulary section (now in weights) — test
2. Remove the identifiability rules (now in weights) — test
3. Remove the stage workflow (now in weights) — test
4. Keep only the role definition (3-5 sentences)

The result is a small model (8B, runs on a laptop) with a minimal prompt that
matches the quality of the original 70B + full system prompt.

### What the system prompt becomes after fine-tuning

```
You are a materials characterization assistant for composite micromechanics.
You help material suppliers characterize fiber-reinforced composites using
inverse estimation and forward prediction tools.
Always verify that the user's measurements are sufficient to identify the
unknowns they are asking about before running any computation.
```

---

## 11. Hardware Requirements

### For deployment (inference only)

Material suppliers need hardware capable of running the local model. The agent
itself (tool file, service layer, database) is lightweight — the constraint is
the language model.

**Before fine-tuning:**

| Hardware | Model | Notes |
|---|---|---|
| Workstation, 32 GB RAM | Llama 3.3 32B Q4 | ~20 tokens/sec. Development baseline. 70B does not fit. |
| Workstation, 32 GB RAM | Qwen2.5 32B Q4 | Comparable speed and quality. Worth benchmarking against Llama 3.3 32B. |
| Workstation, 64 GB RAM | Llama 3.3 70B Q4 | ~25 tokens/sec. Best pre-fine-tune quality. |
| Workstation + RTX 4090 (24 GB VRAM) | Qwen2.5 72B Q4 | ~60 tokens/sec with GPU offload. |
| MacBook Pro M2/M3 Max (64 GB unified) | Llama 3.3 70B Q4 | ~15 tokens/sec. Workable. |

**After fine-tuning (8B model):**

| Hardware | Model | Notes |
|---|---|---|
| Laptop, 16 GB RAM | Fine-tuned Llama 3.1 8B Q4 | ~30 tokens/sec. Runs on standard engineering laptop. |
| MacBook Pro M2 (16 GB) | Fine-tuned 8B Q4 | Fully viable. |

### For fine-tuning (developer only, not needed at supplier site)

| Task | Hardware |
|---|---|
| QLoRA fine-tuning, 8B model, 2000 examples | RTX 4090 (24 GB), ~6-8 hours |
| QLoRA fine-tuning, 13B model, 2000 examples | 2× RTX 4090 or A100 (40 GB), ~12 hours |
| Synthetic data generation (teacher model) | Cloud API or large local model |

Fine-tuning is a one-time developer activity. The resulting LoRA adapter is
distributed to supplier sites as a small file alongside the base model.

---

## Summary

| File | Job | Primary investment |
|---|---|---|
| `vocabulary.md` | Field name mapping, units, synonyms | Write once, extend as needed |
| `agent_tools.py` | Thin wrappers calling service layer | Low effort — service layer is already clean |
| `system_prompt.md` | Physics knowledge, identifiability rules, workflow | Highest effort — this is where correctness lives |

The system prompt is the right place to start. It determines whether the model
can reason correctly about composite characterization. The tool file and
vocabulary follow naturally once the system prompt establishes what the agent
needs to know and do.

After the system prompt is working well, every conversation becomes training
data. The fine-tuning roadmap describes how to eventually distill that
knowledge into a small model that runs on a laptop with no system prompt beyond
a role definition.

---

## 12. LangGraph — State Tracking and Routing

### Why LangGraph (and where it fits)

There is a tension between the model-smart design philosophy and using a graph
framework. LangGraph encodes control flow in the graph — which, if overused,
shifts reasoning responsibility away from the model and back into code.

LangGraph is used here only for what a plain ReAct loop cannot provide:

1. **Persistent state across turns.** The four-stage workflow is inherently
   stateful. Stage 2 must inherit Stage 1's microstructure and matrix modulus
   as fixed inputs. Without a state object, the model must re-derive the current
   stage from conversation history on every turn — fragile for a 32B local model.

2. **SQLite checkpointing.** Users leave conversations and return. The full
   agent state must survive process restarts. See Section 13.

Everything else — intent classification, identifiability reasoning, clarifying
questions, unit conversion, result explanation — stays in the model via the
system prompt. The graph does not route based on physics.

---

### The state object

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages:         Annotated[list[BaseMessage], add_messages]
    current_card_id:  int | None   # which card we are working on
    completed_stages: list[str]    # e.g. ["elastic", "thermoelastic"]
```

`locked_inputs` is not stored in state. The service layer reads prior stage
results directly from the DB via `card_id` when a solver tool is called — the
same way the GUI loads a card before running the next stage. This requires the
user to save after each stage before proceeding. The model enforces this
conversationally.

`completed_stages` is the load-bearing field. It is injected into the model's
system prompt on every turn so the model knows what has been done and what
stage to suggest next — without making a DB call each turn.

---

### The three nodes

The entire graph has three nodes. No more are needed.

**`agent_node`** — the only LLM call. All intelligence lives here. The model
sees all tools, the full system prompt, and the current state context:

```python
def agent_node(state: AgentState):
    context = f"""
Current material card: {state['current_card_id'] or 'none loaded'}
Completed stages: {state['completed_stages'] or 'none'}
"""
    messages = [SystemMessage(SYSTEM_PROMPT + context)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

**`tool_executor`** — LangGraph's built-in `ToolNode`. Zero custom code.
Executes whatever tool call the model returned:

```python
tools = [
    run_forward,
    run_elastic_inverse,
    run_thermoelastic_inverse,
    run_thermal_inverse,
    run_transfer,
    load_material_card,
    save_to_card,
    list_materials,
    list_cards,
    get_card_status,
    get_model_inputs,
    get_model_outputs,
]
tool_executor = ToolNode(tools)
```

**`stage_updater`** — pure Python, no LLM. The only graph-level logic. Runs
after every tool execution. If the tool that just ran was a stage-completing
solver, it updates `completed_stages` and extracts `locked_inputs` from the
result. All other tools leave state unchanged:

```python
def stage_updater(state: AgentState):
    last_msg = state["messages"][-1]   # ToolMessage from tool_executor
    result   = json.loads(last_msg.content)

    # Stage completions — replace, not append, to handle re-runs cleanly
    DOWNSTREAM = {
        "elastic":       ["thermoelastic", "thermal"],
        "thermoelastic": ["thermal"],
        "thermal":       [],
    }

    if last_msg.name == "run_elastic_inverse" and result.get("success"):
        stages = [s for s in state["completed_stages"]
                  if s not in DOWNSTREAM["elastic"] and s != "elastic"]
        return {
            "completed_stages": stages + ["elastic"],
            "current_card_id":  result["card_id"],
        }

    if last_msg.name == "run_thermoelastic_inverse" and result.get("success"):
        stages = [s for s in state["completed_stages"]
                  if s not in DOWNSTREAM["thermoelastic"] and s != "thermoelastic"]
        return {"completed_stages": stages + ["thermoelastic"]}

    if last_msg.name == "run_thermal_inverse" and result.get("success"):
        stages = [s for s in state["completed_stages"] if s != "thermal"]
        return {"completed_stages": stages + ["thermal"]}

    if last_msg.name == "load_material_card" and result.get("success"):
        return {"current_card_id": result["card_id"]}

    return {}  # all other tools — state unchanged
```

---

### Graph wiring

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

graph = StateGraph(AgentState)

graph.add_node("agent",         agent_node)
graph.add_node("tools",         tool_executor)
graph.add_node("stage_updater", stage_updater)

graph.set_entry_point("agent")

# After the model responds: did it make a tool call?
graph.add_conditional_edges(
    "agent",
    lambda s: "tools" if s["messages"][-1].tool_calls else END,
)

# After tool execution: always update state, then back to agent
graph.add_edge("tools",         "stage_updater")
graph.add_edge("stage_updater", "agent")

checkpointer = MemorySaver()   # in-memory only — state is restored from DB on resume
app = graph.compile(checkpointer=checkpointer)
```

That is the complete graph. ~15 lines of wiring. The routing, the
identifiability reasoning, the unit checking, the clarifying questions — all
of that stays in the model via the system prompt.

---

### Visual diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                            AgentState                                 │
│   messages | current_card_id | completed_stages | locked_inputs       │
└──────────────────────────────────────────────────────────────────────┘
                              │ injected into system prompt each turn
                              │
              ┌───────────────▼───────────────┐
              │           agent_node           │◄──────────────────┐
              │       (LLM — all logic)        │                   │
              │                                │                   │
              │  • reads state context         │                   │
              │  • maps language → tool call   │                   │
              │  • decides when NOT to call    │                   │
              │  • explains results to user    │                   │
              └───────────────┬───────────────┘                   │
                              │                                    │
              ┌───────────────┴───────────────┐                   │
         tool_calls?                     no tool_calls             │
              │                               │                    │
              ▼                             [END]                  │
┌─────────────────────────┐                                        │
│      tool_executor      │                                        │
│   (ToolNode — no logic) │                                        │
│                         │                                        │
│  run_forward            │                                        │
│  run_elastic_inverse    │                                        │
│  run_thermoelastic_inv  │                                        │
│  run_thermal_inverse    │                                        │
│  run_transfer           │                                        │
│  load_material_card     │                                        │
│  save_to_card           │                                        │
│  list_materials         │                                        │
│  list_cards             │                                        │
│  get_card_status        │                                        │
│  get_model_inputs       │                                        │
│  get_model_outputs      │                                        │
└─────────────┬───────────┘                                        │
              │                                                    │
              ▼                                                    │
┌─────────────────────────────────────────────┐                   │
│               stage_updater                 │                   │
│           (pure Python — no LLM)            │                   │
│                                             │                   │
│  elastic_inverse succeeded?                 │                   │
│    → completed_stages += ["elastic"]        │                   │
│    → locked_inputs ← a11, a22, a12, a13,   │                   │
│                       a23, fiber_massfrac,  │                   │
│                       ar, matrix_E,         │                   │
│                       matrix_nu             │                   │
│                                             │                   │
│  thermoelastic_inverse succeeded?           │                   │
│    → completed_stages += ["thermoelastic"]  │                   │
│    → locked_inputs ← f_cte1, f_cte2, m_cte │                   │
│                                             │                   │
│  thermal_inverse succeeded?                 │                   │
│    → completed_stages += ["thermal"]        │                   │
│    → locked_inputs ← k_p1, k_p2, k_l2, k_t │                   │
│                                             │                   │
│  load_material_card succeeded?              │                   │
│    → current_card_id ← card_id             │                   │
│                                             │                   │
│  anything else → state unchanged           │                   │
└─────────────────────────────────────────────┘                   │
              │                                                    │
              └────────────────────────────────────────────────────┘
```

---

### What the graph deliberately does not encode

- Intent classification (forward vs inverse vs transfer) — model decides
- Whether measurements are sufficient to run a solve — model decides
- Which fields are free vs fixed per stage — model decides from system prompt
- Unit conversion — model decides
- Whether to ask a clarifying question — model decides

If any of these decisions were encoded as graph edges, the design would drift
towards tool-smart behavior. The graph is a state carrier and a loop, not a
decision tree.

---

## 14. Intent Routing — Mode-Based Tool Filtering

### The problem

Small local models (≤14B) fail at intent classification when all 16 tools are
visible simultaneously. Given a prediction request, the model may still reach
for an inverse solver. Given a database query, it may call `predict_properties`.
Prompt rules help but are unreliable — the model reads past them.

The root cause: asking a small model to (1) classify intent, (2) select tools,
(3) handle physics reasoning, and (4) format the response — all in one pass —
is too much. Intent routing offloads step 1 to the graph so the model never
faces the choice.

---

### Approach: graph-level mode routing with per-message re-classification

A lightweight `mode_router` node runs before every LLM call. It reads the
user's message, classifies the intent, and passes only the relevant tool subset
to the model. The model cannot call tools outside its assigned mode because
they are not in its bound list for that turn.

Modes are **per-message, not per-session** — the router re-classifies every
new human message. This means the user can switch naturally between prediction,
inverse, and database queries within the same conversation without mode-lock.

---

### Two router implementations

**Option A — Keyword prefix (explicit, zero latency)**

User types a prefix: `PREDICT:`, `INVERSE:`, `SEARCH:`, `CARD:`, `DATABASE:`,
`SAVE:`. The router does a string match. No model call, <1ms.

Tradeoff: requires user training. Good for power users or developers. Better
than nothing while the BERT router is being built.

**Option B — Mini NLI classifier (transparent to user, ~20-50ms)**

A tiny zero-shot NLI model (e.g. `cross-encoder/nli-MiniLM2-L6-H4`, 67 MB)
scores the user message against each mode's description and picks the highest.
The user types naturally. Runs fully offline on CPU in <50ms on M2 Max.

This is the target architecture. Both options share the same downstream graph
structure — only the `mode_router` node internals differ.

---

### Mode → tool mapping

| Mode | Primary use | Tools available |
|---|---|---|
| `predict` | Forward property prediction | `predict_properties`, `predict_thermal_conductivity`, `convert_fraction`, `get_material_details`, `list_materials`, `get_card_status`, `list_cards` |
| `inverse` | Elastic / thermoelastic / thermal inverse solvers | `run_elastic_inverse`, `run_thermoelastic_inverse`, `run_thermal_inverse`, `check_identifiability`, `convert_fraction`, `get_material_details`, `list_materials`, `get_card_status`, `list_cards`, `save_to_card`, `save_processing_conditions` |
| `search` | Knowledge base queries | `search_knowledge_base` |
| `predict_search` | Predict + knowledge base together | predict tools + `search_knowledge_base` |
| `card` | Inspect and browse saved material cards | `list_cards`, `get_card_status`, `inspect_card_inputs`, `get_model_inputs_outputs`, `get_material_details` |
| `database` | Add or query materials in the library | `add_fiber`, `add_polymer`, `list_materials`, `get_material_details` |
| `save` | Save results after a solve | `save_to_card`, `save_processing_conditions`, `get_card_status`, `list_cards` |
| `default` | No mode detected — backward compatible | all tools |

**Bridge tools** appear in multiple modes because they are needed mid-workflow
regardless of the primary intent. `get_card_status` and `list_cards` appear in
`predict`, `inverse`, `card`, and `save` — you often need to look up a card
before or after any operation. `save_to_card` appears in both `inverse` and
`save` so you can save immediately after a solve without mode-switching.

---

### Updated graph structure

```
START
  │
  ▼
mode_router          ← per-message: classifies intent, writes active_mode to state
  │                    pure Python (keyword match or NLI model) — no LLM call
  ▼
agent_node           ← LLM bound to mode-specific tool subset only
  │
  ├─ tool calls? ─YES─→ tool_executor ─→ stage_updater ─→ agent_node (loop)
  │
  └─ NO ──────────────────────────────────────────────────→ END
```

`mode_router` is a pure Python node — zero LLM calls.
`tool_executor` uses the full TOOLS list (the LLM can only call what it was
bound to, so restriction at bind time is sufficient).
`stage_updater` is unchanged from Section 12.

---

### State change

One field added to `AgentState`:

```python
active_mode: str  # "predict" | "inverse" | "search" | "predict_search" |
                  # "card" | "database" | "save" | "default"
```

---

### Implementation status

Not yet implemented. The tool set must be complete before the routing map is
finalised — adding `run_thermoelastic_inverse` and `run_thermal_inverse` to the
TOOLS list will affect which tools belong in the `inverse` mode. See the build
order note below.

---

### Recommended build order

**Finish core tools first, then wire the router.**

The `inverse` mode tool list is currently incomplete — `run_thermoelastic_inverse`
and `run_thermal_inverse` are implemented in `core/` but not yet exposed in
`agent_tools.py`. Adding them after the router is built means revisiting the
mode map, the NLI label descriptions, and any mode-specific prompt injections.

Correct order:
1. Add `run_thermoelastic_inverse` to `agent_tools.py` (Stage 2 — CTE)
2. Add `run_thermal_inverse` to `agent_tools.py` (Stage 3 — thermal conductivity)
3. Finalise the `inverse` mode tool list
4. Implement `mode_router` (keyword first, NLI second)
5. Add `active_mode` to `AgentState`
6. Update `agent_node` to use pre-bound LLM variants per mode

---

## 13. Session Persistence and Memory Restoration

### Why not SqliteSaver

LangGraph's `SqliteSaver` checkpointer writes the full `AgentState` at every
node transition — including the entire `messages` list each time. Because
`messages` grows with every turn, storage grows quadratically: a 50-turn
conversation produces 150 checkpoint rows, each re-serializing the full
growing history. For a desktop app with a single SQLite file, this is wasteful
and unnecessary.

More importantly, the chat transcript is not what needs to survive between
sessions. What matters is the **structured state**: which stages are complete
and what values were locked in. That information already lives in the existing
DB tables.

### Working state vs committed state (Unit of Work pattern)

There are two distinct states per stage:

| State | What it means | Where it lives |
|---|---|---|
| **Run but not saved** | Solver completed this session, user has not saved | `MemorySaver` only — lost on restart |
| **Saved** | User explicitly called "Save to Card" | `constituent_property_values` + `microstructure_snapshots` in DB |

`inference_runs` is an audit trail written only on explicit save — it is not
queried for restore. On resume, only committed (saved) state is available.
Unsaved in-session results are lost on restart, exactly as in the GUI.

### What to persist and where it already lives

| State field | Where it lives after "Save to Card" |
|---|---|
| `completed_stages` | Inferred from presence of stage-specific properties in `constituent_property_values` |
| `current_card_id` | `print_configs` — identified by card name |
| `messages` | Not persisted — not needed |
| Prior stage results (microstructure, matrix props, CTEs, k params) | In DB — read by service layer directly when next solver tool is called |

`locked_inputs` is not in state. The service layer reads prior stage results
from the DB via `card_id` when a solver tool is called — the same way the GUI
loads a card before running Stage 2. The user must save after each stage before
proceeding. The model enforces this conversationally.

No new tables. No checkpoint blobs. The DB stays clean.

### Restoring completed stages from the DB

`completed_stages` is inferred from which stage-specific properties exist in
`constituent_property_values` for the card:

```python
def restore_state_from_card(card_id: int) -> AgentState:
    props  = get_constituent_property_values(card_id)  # existing db.py helper
    stages = []

    if "matrix_E" in props:
        stages.append("elastic")
    if "f_cte1" in props:
        stages.append("thermoelastic")
    if "k_p1" in props:
        stages.append("thermal")

    return AgentState(
        messages         = [],
        current_card_id  = card_id,
        completed_stages = stages,
    )

# On resume
memory = MemorySaver()
app    = graph.compile(checkpointer=memory)
state  = restore_state_from_card(card_id)
app.invoke({"messages": [HumanMessage("ok I have the CTE data now")]},
           config={"configurable": {"thread_id": f"card_{card_id}"}})
```

The model's first response continues naturally because its system prompt
already contains:

```
Current card: 3
Completed stages: ["elastic"]
```

It has everything it needs without the transcript.

### Why discarding the transcript is fine

The transcript records what was *said*. The structured state records what was
*learned*. For this domain, only the latter matters between sessions:

- The user does not need to re-explain their material system
- The model does not need to re-ask for Stage 1 results
- Any measurements the user provided are already saved in `experimental_measurements`

If the user wants to review what was previously discussed, `get_card_status()`
gives them a complete summary derived from the DB — more reliable than
reconstructed chat history.

### Finding the card on resume

```python
# At session start — resolve card before entering the graph
user_input = "continue my Carbon T300 / PESU work"
card_id    = resolve_card_from_message(user_input)  # simple name lookup, not an LLM call

if card_id is not None:
    state = restore_state_from_card(card_id)   # reconstruct from DB
else:
    state = AgentState(messages=[], current_card_id=None,
                       completed_stages=[], locked_inputs={})

app.invoke({"messages": [HumanMessage(user_input)]},
           {"configurable": {"thread_id": f"card_{card_id or 'new'}"}})
```
