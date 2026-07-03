# DiffMicromechanics — Full System Context

This document describes a software system for automated characterization of fiber-reinforced
composite materials. It is intended as context for writing a research paper or commercial
pitch. Read this fully before generating any text.

---

## What the system does in one sentence

Given a small set of experimentally measured composite properties (stiffness, thermal
expansion, thermal conductivity), the system automatically recovers the underlying
**constituent material properties** and **microstructure** that produced them — and uses
those inferred values to predict composite behavior on any new printer or process condition,
without additional testing.

---

## The core problem it solves

Fiber-reinforced composites manufactured by additive manufacturing (AM) — such as
continuous fiber FDM, short-fiber FDM, or automated fiber placement — have properties
that depend on two fundamentally different layers:

1. **Constituent properties** — intrinsic to the fiber and matrix materials themselves.
   Examples: matrix Young's modulus, fiber axial/transverse CTE, fiber thermal
   conductivity. These do not change when you switch printers.

2. **Microstructure** — set by the printing process. Examples: fiber orientation tensor
   (a11, a22, …), fiber mass fraction, fiber aspect ratio. These change when you switch
   printers, change nozzle diameter, or change print speed.

3. **Composite properties** — what you measure on a coupon. These are a function of both
   layers (E1, CTE11, K11, …).

**The problem:** Standard composite characterization treats composite-level measurements
as end-results. There is no systematic method to decompose them back into constituent
properties and microstructure — and without that decomposition, you cannot transfer a
material characterization from one printer to another.

**What this system does:** It inverts the composite structure–property relationship,
recovering constituent properties and microstructure from composite measurements.
Because constituent properties are printer-independent, a single characterization campaign
on Printer A provides enough information to predict composite properties on Printer B
using only the new printer's orientation tensor — no additional coupon testing required.

---

## The four-stage characterization workflow

### Stage 1 — Elastic inverse

**Goal:** From measured composite elastic properties (E1, E2, E3, G12, G13, G23, nu12…),
infer the microstructure (fiber orientation tensor a11, a22; fiber mass fraction; fiber
aspect ratio) and the in-situ matrix modulus.

**Why it matters:** The matrix modulus in a printed composite differs from the neat
polymer datasheet value because the printing process introduces voids, residual stress,
and incomplete crystallization. This stage recovers the true in-situ matrix modulus, not
a textbook value.

**Free variables:** a11, a22, fiber_massfrac, aspect_ratio, matrix_modulus,
matrix_poisson (if shear measurements are available).

**Fixed variables:** All fiber elastic properties from the datasheet (E1, E2, G12,
nu12, nu23, density).

**Output:** A complete microstructure snapshot + in-situ matrix modulus, saved to the
material card with "inferred" provenance.

---

### Stage 2 — Thermoelastic inverse

**Goal:** From measured composite CTEs (CTE11, CTE22), infer the fiber axial CTE
(f_cte1), fiber transverse CTE (f_cte2), and matrix CTE (m_cte).

**Why it matters:** Fiber CTE values in datasheets are often measured on tows, not
in-composite. The transverse fiber CTE (f_cte2) is rarely published and is critical for
out-of-plane thermal distortion prediction. This stage recovers both CTE components
directly from composite-level measurements.

**Fixed variables:** Everything inferred in Stage 1 is held fixed here. This is the key
staging principle — once you know the microstructure, CTE identification becomes a
well-posed problem with only 3 unknowns and 2–3 measurements.

**Output:** f_cte1, f_cte2, m_cte stored **globally** against the fiber/polymer pair —
not against the card — because they are intrinsic material properties reusable for any
printer that uses the same materials.

---

### Stage 3 — Thermal inverse

**Goal:** From measured composite thermal conductivity K vs temperature (K11(T),
K22(T), K33(T)), infer the fiber conductivities (k_f1, k_f2) and a parametric
temperature-dependent polymer conductivity model: k_m(T) = p1·√T + p2.

**Why it matters:** Fiber thermal conductivities (especially the transverse k_f2) are
nearly impossible to measure directly on fibers. The composite-level K vs T signature
carries enough information to recover them indirectly. The parametric polymer model
captures the temperature dependence of the matrix conductivity with just two scalars.

**Output:** k_f1, k_f2, p1, p2 stored globally against the fiber/polymer pair.

---

### Stage 4 — Transfer to new printer (forward prediction)

**Goal:** Given a new printer's orientation tensor (measured by CT, ultrasound, or
process simulation), predict the full set of composite elastic, thermoelastic, and
thermal properties without any additional coupon testing.

**Why it matters:** This is the commercial value proposition. A single characterization
campaign on a reference printer provides all constituent properties. The new printer
only needs its orientation tensor — which can be measured from a small CT scan or
estimated from process simulation. The full composite property prediction (12+ outputs)
follows immediately from the surrogate model.

---

## The surrogate model architecture

Each of the three physical domains (elastic, thermoelastic, thermal) has a dedicated
neural network surrogate model trained to approximate the full composite micromechanics
relationship:

```
inputs (fiber props + matrix props + microstructure) → surrogate → composite properties
```

**Why surrogates instead of analytical models (Halpin-Tsai, Mori-Tanaka, etc.)?**
- Analytical micromechanics models assume simplified fiber geometry and fail for
  arbitrary orientation distributions.
- The orientation averaging approach (using the second-order orientation tensor a_ij)
  requires expensive numerical integration over all orientations.
- The surrogate directly maps orientation tensor components to composite properties,
  bypassing the orientation averaging entirely.
- Inference speed: a single surrogate evaluation takes microseconds vs. milliseconds
  for orientation averaging, enabling thousands of optimizer evaluations per second.

**Training data:** Generated from a high-fidelity micromechanics model using
Latin hypercube sampling over the full input space (fiber/matrix property ranges,
orientation tensor components, fiber mass fraction, aspect ratio).

---

## The inverse solver

Each stage uses gradient-based optimization (L-BFGS-B) with:

- **Epsilon-insensitive loss:** When measurement uncertainties (sigma values) are
  provided, residuals inside ±sigma contribute zero loss. This correctly handles
  experimental scatter — measurements with larger uncertainty exert less pull on
  the solution.

- **Multi-restart:** The optimizer is run from multiple random starting points to
  escape local minima. The best result across restarts is returned.

- **Bounds:** All free variables are constrained to physically meaningful ranges
  (e.g., a11 ∈ [0.5, 0.85], matrix_modulus ∈ [2000, 5000] MPa).

- **Fit quality assessment:** The normalized fit error is reported and classified
  (good < 0.01, acceptable < 0.05, poor > 0.05). A quality checker node in the agent
  graph evaluates whether the fit is trustworthy before recommending saving.

---

## Identifiability analysis (Fisher Information Matrix)

Before running any inverse stage, the system can assess whether the available
measurements contain enough information to identify the unknowns. This uses the
**Fisher Information Matrix (FIM)** computed numerically by evaluating the surrogate
Jacobian over a sample of the parameter space.

**Output per free variable:**
- WELL determined — Cramér-Rao lower bound on standard deviation is small relative
  to the parameter range.
- MARGINAL — identifiable but with significant uncertainty.
- POOR — the measurement set does not constrain this parameter.

**Practical value:** Prevents wasted experimental campaigns. For example, FIM analysis
immediately reveals that matrix_poisson is not identifiable from E1/E2/E3 alone —
at least one shear measurement (G12 or nu12) is required. This is shown before the
user runs any experiments.

**Recommended additional measurements:** The FIM analysis ranks which additional
measurements would most improve identifiability, allowing targeted experimental planning.

---

## The agent interface

The system is accessible through a conversational LLM agent built with LangChain and
LangGraph. The agent has 21 tools covering the full workflow:

| Tool | Purpose |
|---|---|
| `list_materials` | Browse the fiber/polymer/printer library |
| `get_material_details` | Full datasheet for one material |
| `convert_fraction` | Mass fraction ↔ volume fraction |
| `check_identifiability` | FIM analysis |
| `run_elastic_inverse` | Stage 1 solver |
| `run_thermoelastic_inverse` | Stage 2 solver |
| `run_thermal_inverse` | Stage 3 solver |
| `run_full_pipeline` | All 3 stages from a single Excel file |
| `predict_properties` | Forward prediction (elastic + thermoelastic) |
| `predict_thermal_conductivity` | Forward prediction (thermal, with T range) |
| `sweep_parameter` | What-if parameter sweeps |
| `save_to_card` | Persist results with provenance |
| `delete_card` | Remove a card and all its data |
| `get_card_status` | Full view of a material card |
| `inspect_card_inputs` | Preview what the forward model will use |
| `list_cards` | All characterized material cards |
| `add_fiber` / `add_polymer` | Extend the material library |
| `search_knowledge_base` | RAG over uploaded composites literature |
| `save_processing_conditions` | Record print settings per card |

**Agent architecture:** LangGraph three-node graph — `agent → tool_executor → checker
→ agent`. The checker node appends a PASS/FAIL quality report to every inverse solver
result before the LLM reads it. This prevents the agent from recommending saving a
poor-fit result without flagging it to the user.

---

## Privacy-preserving design (local LLM)

The agent is designed to run entirely offline using **Qwen 2.5 14B via Ollama** on
local hardware. No data, measurements, or material properties leave the user's machine.

**Why this matters for aerospace and defence-adjacent manufacturing:**
- Composite material properties are often proprietary or export-controlled (ITAR/EAR).
- Even cloud providers with enterprise privacy contracts do not fully satisfy ITAR
  requirements — the contract covers data storage, not inference.
- A 14B-parameter quantized model running on a local workstation (or A100 GPU) provides
  the full agent capability without any external API calls.
- The system is also compatible with frontier models (Claude, GPT-4) as a drop-in
  replacement for users without data privacy constraints.

**The "automation intelligence before artificial intelligence" thesis:** The agent does
not need to be highly capable at reasoning — it needs to reliably route user intent to
the correct tool sequence. A 14B local model is sufficient for this task. The scientific
intelligence lives in the surrogate models and the staged inverse framework, not in the
LLM. This argues for a class of domain-specific engineering assistants where small local
models are architecturally appropriate.

---

## The material card and provenance system

Every inferred or predicted value is stored with a **provenance tag**:
- `web` — value from a public database or datasheet
- `inputted` — value entered by the user (e.g., fiber E1 from the datasheet)
- `inferred` — value recovered by the inverse solver
- `predicted` — value output by the forward surrogate

The **material card** (called `print_config` in the database) represents a
(fiber, polymer, printer) triple. Each card accumulates results from all four stages.
The card structure enforces the physical separation:
- Constituent properties are stored **globally** (shared across all cards using the
  same fiber and polymer) — a single characterization reuses across all printers.
- Microstructure is stored **per card** — it is printer-specific.
- Composite properties are stored **per card**.

This design directly encodes the physics: you characterize materials once; you
characterize microstructure per printer.

---

## Key advantages over existing practice

| Aspect | Current practice | This system |
|---|---|---|
| Characterization approach | Rule-of-mixtures / datasheet look-up | Inverse identification from composite measurements |
| Matrix modulus | Neat polymer datasheet | In-situ value recovered from printed coupon |
| Fiber CTE | Axial only, from tow test | Both axial and transverse, recovered from composite |
| Thermal conductivity | Room temperature, isotropic | Temperature-dependent, anisotropic, recovered from composite |
| Transfer to new printer | Re-run full characterization | Constituent properties carry over; only orientation needed |
| Measurement planning | Trial and error | FIM identifiability analysis before experiments |
| Identifiability | Ignored | Explicitly assessed per variable per measurement set |
| Provenance | None — values copied from datasheets | Every value tagged: inferred / inputted / measured / predicted |
| Software interface | Spreadsheets / custom scripts | Conversational agent, runs locally |
| Privacy | Data sent to cloud APIs | Entire pipeline runs offline |

---

## Target applications

**Aerospace (primary):**
- Qualification of additively manufactured composite structures. The staged
  characterization workflow directly maps to the building-block approach (coupon →
  element → subcomponent → component) used in aerospace certification.
- Process transfer between certified and non-certified printers. Constituent properties
  from a certified Printer A can be used to bound predicted properties on Printer B,
  reducing the test matrix required for certification.
- Design allowables generation with uncertainty quantification via the FIM.

**General manufacturing:**
- Supplier qualification: characterize a fiber/matrix system once, predict composite
  properties across any orientation distribution.
- Process optimization: sweep orientation tensor and mass fraction to find the process
  window that meets stiffness or CTE targets.
- Multi-material selection: compare predicted composite properties across fiber/polymer
  combinations without physical testing.

---

## What is novel (paper claims)

1. **Staged inverse framework with constituent/microstructure separation** — the specific
   three-stage sequence (elastic → thermoelastic → thermal) that progressively fixes
   inferred quantities and reduces the inverse problem dimensionality at each stage.

2. **Recovery of transverse fiber CTE from composite measurements** — f_cte2 is not
   available in any standard fiber datasheet; this system recovers it from standard
   composite CTE measurements via the thermoelastic inverse.

3. **Temperature-dependent thermal constituent recovery** — recovering the parametric
   polymer conductivity model k_m(T) = p1·√T + p2 from composite K vs T data, using a
   surrogate-accelerated inverse.

4. **FIM-based identifiability analysis for composite inverse problems** — pre-experiment
   assessment of whether a given measurement set can identify a given set of unknowns,
   with ranked recommendations for additional measurements.

5. **Agent-based interface with staged quality gating** — the checker node that
   intercepts solver results before the LLM reads them, preventing the agent from
   silently accepting poor-fit results.

6. **Privacy-preserving engineering agent** — full four-stage workflow accessible
   through a local LLM, with no IP leaving the user's hardware.

---

## What to compare against in related work

When positioning this paper, the key literature clusters are:

- **Virtual assistants for simulation software** (Abaqus, OpenFOAM, ANSYS Mechanical):
  these use LLMs to automate simulation setup and postprocessing. They do NOT solve
  inverse problems and do not separate constituent from microstructure properties.
  They are fundamentally different: scripting assistants vs. characterization systems.

- **Surrogate models for composites** (physics-informed NNs, data-driven
  micromechanics): these focus on forward prediction accuracy. They do not address the
  inverse problem or the identifiability of constituent properties.

- **Inverse methods in composites** (neural network fitting, genetic algorithm
  identification): these typically recover a single property set from elastic data.
  They do not address thermoelastic or thermal stages, do not separate constituent from
  microstructure, and do not address process transfer.

- **Digital twin / process–structure–property chains**: these address process parameters
  → microstructure → properties. This system occupies the microstructure → properties
  → constituent identification leg of that chain, which is the least developed.

---

## Limitations (honest assessment for paper)

- **Surrogate accuracy bounds the inverse accuracy.** If the surrogate has a mean
  prediction error of 2% on elastic outputs, the inverse solver cannot achieve better
  than ~2% accuracy on inferred properties.

- **Orientation tensor completeness.** The second-order orientation tensor (5 independent
  components) is a sufficient statistic for the surrogate but does not capture
  orientation distribution shape (e.g., bimodal distributions may have the same a_ij
  as a unimodal one).

- **In-situ vs neat constituent properties.** The system recovers in-situ properties
  (as they exist inside the composite). These may differ from neat properties due to
  interface effects, sizing, void content, etc. The system does not decompose this
  further — it is a limitation and also a feature (in-situ is what matters for
  performance prediction).

- **Stage 3 requires K vs T data over a sufficient temperature range (>50°C).** With
  a narrow temperature range, the polymer conductivity model (p1, p2) is poorly
  identified. FIM analysis will flag this.

- **Agent reliability depends on the local LLM.** Qwen 2.5 14B occasionally violates
  routing rules (confirmation loops, double tool calls). The system prompt and graph
  checker mitigate but do not fully prevent these.
