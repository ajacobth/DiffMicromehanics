# DiffMicromechanics

A tool for characterising and predicting properties of fiber-reinforced
composites. Given lab measurements from a printed sample, it recovers the
underlying constituent material properties (matrix stiffness, fiber/matrix CTE,
thermal conductivities) and uses them to predict how the same materials will
behave on a different printer — without running new experiments.

---

## Two ways to use it

### MateriAl — conversational AI agent (recommended)

Describe your measurements in plain English. The agent runs the correct solvers,
checks results for physical plausibility, and saves everything to a material
card. No GUI operation or Python knowledge needed.

```
You:    I have a T300/PESU composite on CAMRI. E1 = 15.1 GPa, E2 = 5.2 GPa,
        G12 = 2.4 GPa. Fiber mass fraction is 0.20.

Agent:  [runs elastic inverse solver]
        fit_error = 0.008  (excellent)
        a11 = 0.74, a22 = 0.14, ar = 18.2, matrix_modulus = 3820 MPa
        Save to card?
```

**-> Full setup guide: [final/AGENT_SETUP.md](final/AGENT_SETUP.md)**

### GUIs — graphical interface

Tkinter desktop applications for forward prediction, inverse design, thermal
inverse, and physics-informed transfer to new printers.

**-> Full setup guide: [SETUP_AND_RUN.md](SETUP_AND_RUN.md)**

---

## Quick start (after environment setup)

```bash
# 1. Clone and enter the project
git clone <repository-url>
cd DiffMicromehanics/final

# 2. Activate the conda environment (create it first — see SETUP_AND_RUN.md)
conda activate jax_trial

# 3. Run the one-time setup script
bash setup.sh

# 4. Start the agent
streamlit run app_chat.py

# OR start the GUI
python gui.py
```

> **New user?** Start with [SETUP_AND_RUN.md](SETUP_AND_RUN.md) to create
> the conda environment, then come back here for step 3 onwards.

---

## What it does

Composite material properties depend on three layers:

| Layer | What it is | Printer-dependent? |
|---|---|---|
| **Constituent** | Fiber and matrix stiffness, CTE, conductivity | No — intrinsic to material chemistry |
| **Microstructure** | Fiber orientation tensor, aspect ratio, mass fraction | Yes — set by the printer |
| **Composite** | E1, CTE11, K11, … | Yes — derived from both layers |

The key insight: **constituent properties are printer-independent.** Characterise
once on Printer A, predict for any number of other printers using the same
materials.

### Four capabilities

| Capability | What you provide | What you get |
|---|---|---|
| **Forward prediction** | Constituent inputs + microstructure | Predicted composite properties |
| **Elastic/thermoelastic inverse** | Measured E, G, CTE | Microstructure + constituent properties |
| **Thermal inverse** | Measured K vs T data | Fiber and matrix conductivities |
| **Transfer to new printer** | Source card + new printer's measurements | Full property predictions for new printer |

---

## Documentation

| File | Contents |
|---|---|
| [SETUP_AND_RUN.md](SETUP_AND_RUN.md) | Full environment setup for Windows, Mac, Linux (GUI + agent) |
| [final/AGENT_SETUP.md](final/AGENT_SETUP.md) | Agent-specific setup: Ollama, model selection, knowledge base, API keys |
| [final/README.md](final/README.md) | Technical reference: surrogate models, material card system, database design |

---

## Seeded material library

**Fibers:** Carbon Fiber T300 (Toray), E-Glass (Owens Corning), AS4 (Hexcel)

**Polymers:** PESU Ultrason (BASF), Epoxy 3501-6 (Hexcel)

Add new materials via the agent (`add a fiber called...`) or the GUI
(**Manage Materials**).
