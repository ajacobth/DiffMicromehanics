# MateriAl - User Guide and Conversation Examples

---

## Quick Reference (Cheat Sheet)

**Launch**
```
conda activate jax_trial
streamlit run app_chat.py
```

Or terminal-only:
```
python run_agent.py
python run_agent.py --card 5
```

---

**Keywords (optional, but recommended):**

Starting your message with a keyword helps the agent pick the right tools faster.
You do not have to use them - the agent will figure it out from context.

| Keyword | Use for |
|---|---|
| `INVERSE` | Characterizing from measurements |
| `PREDICT` | Predicting or what-if analysis |
| `SEARCH` | Material properties, theory questions |

---

**Most common things to type:**

Check what is available before starting:
```
what materials do we have?
what printers do we have?
what cards do we have?
```

Stage 1 - Elastic characterization:
```
INVERSE characterize T300/PESU on CAMRI. mf=0.20.
E1=15.14 GPa, E2=5.14 GPa, E3=4.14 GPa, G12=2.50 GPa, nu12=0.35
```

Stage 2 - Thermal expansion (need card ID from Stage 1):
```
INVERSE thermoelastic for card 5. CTE11=10 ppm/K, CTE22=50 ppm/K, CTE33=75 ppm/K
```

Stage 3 - Thermal conductivity (need card ID and a CSV file):
```
INVERSE thermal for card 5, CSV at /Users/me/k_vs_T.csv
```

All stages from one CSV file:
```
INVERSE characterize from /Users/me/measurements.csv
```

Save a result:
```
save it as "T300_PESU_CAMRI_Jan2026"
```

Predict from a saved card:
```
PREDICT what are the elastic properties for card 5?
```

What-if sweep:
```
PREDICT sweep ar from 10 to 30 for card 5, show E1
PREDICT how much do I increase mf to get E1=20 GPa for card 5?
```

---

**Units:**

| Property | Unit | Example |
|---|---|---|
| Stiffness (E, G) | GPa | `E1=15.14 GPa` |
| Poisson ratio | dimensionless | `nu12=0.35` |
| Thermal expansion (CTE) | ppm/K | `CTE11=10 ppm/K` |
| Thermal conductivity (K) | W/m.K | in the CSV file |
| Temperature | Celsius | `25, 50, 75, 100` |

---

**Quick fixes:**

| Problem | What to type |
|---|---|
| Forgot card ID | `what cards do we have?` |
| Material not found | `what materials do we have?` |
| Fit error too high | `why is my fit error so high?` |
| Stage 2 failing | Make sure Stage 1 is saved first |

---

*Full walkthrough and example conversations below.*

---

MateriAl is a chat assistant for characterizing fiber-reinforced composite materials.
You describe your material and paste in your lab measurements. The agent runs
micromechanics models to infer the underlying material properties and saves them
to a reusable material card.

---

## Getting Started

Launch the app from the `final/` directory:

```
conda activate jax_trial
streamlit run app_chat.py
```

Or, for a terminal-only session:

```
python run_agent.py
python run_agent.py --card 5    # resume an existing card
```

Before starting a characterization, confirm that your fiber, polymer, and printer
are in the database. If any are missing, the agent will tell you and you can add them.

---

## How a Characterization Works

A full characterization runs up to three stages in order. Each stage builds on
the previous one and adds results to a material card.

| Stage | What you provide | What the agent infers |
|---|---|---|
| 1 - Elastic | E1, E2, G12, nu12 (and optionally E3) | Fiber orientation, fiber mass fraction, aspect ratio, matrix stiffness |
| 2 - Thermal expansion | CTE11, CTE22 (and optionally CTE33) | Fiber CTE (longitudinal and transverse), matrix CTE |
| 3 - Thermal conductivity | K vs temperature CSV | Fiber conductivity (longitudinal and transverse), matrix conductivity model |

You do not have to run all three. Stage 1 is always required. Stages 2 and 3 are
optional and only needed if you have those measurements.

---

## Message Prefixes

Starting your message with a keyword tells the agent which mode to use:

| Prefix | Use for |
|---|---|
| `INVERSE` | Running any characterization stage |
| `PREDICT` | Forward prediction and what-if analysis |
| `SEARCH` | Material property lookups and theory questions |
| *(none)* | General questions - agent decides |

---

## Stage 1 - Elastic Characterization

**What you need:** stiffness measurements from tensile or DMA testing. E1 and E2
are the minimum. Adding G12 and nu12 significantly improves the result. E3 is
a bonus.

**Ideal measurement set:** E1, E2, E3, G12, nu12

**Fit error guide:**

| Value | Interpretation |
|---|---|
| Below 0.01 | Excellent |
| 0.01 to 0.10 | Acceptable |
| Above 0.10 | Measurements may be inconsistent or insufficient - see Troubleshooting |

### Example conversation

```
You:      INVERSE I want to characterize my T300/PESU composite on the CAMRI printer.
          Fiber mass fraction is 0.20.
          Measurements:
            E1  = 15.14 GPa +/- 1.25 GPa
            E2  =  5.14 GPa +/- 0.30 GPa
            E3  =  4.14 GPa +/- 0.20 GPa
            G12 =  2.50 GPa +/- 0.15 GPa
            nu12 = 0.35

Agent:    Ready to run elastic inverse:
            Fiber: T300 Carbon Fiber (Toray)
            Polymer: PESU Ultrason (BASF)
            Printer: CAMRI
            mf fixed at 0.20 | ar, a11, a22, matrix_modulus free
          Confirm? (yes / change something)

You:      yes

Agent:    Elastic inverse complete.

          Inferred microstructure:
            a11 = 0.7896   a22 = 0.1613   ar = 21.0
            fiber_massfrac = 0.20

          Inferred constituent:
            matrix_modulus = 2905 MPa

          Fit error: 0.0031  (excellent)

          Save to a card? (provide a name or say "just save it")

You:      save it as "T300_PESU_CAMRI_Jan2026"

Agent:    Saved. Card #5 - T300_PESU_CAMRI_Jan2026
```

---

## Stage 2 - Thermal Expansion Characterization

**What you need:** CTE measurements from dilatometry. CTE11 and CTE22 are
required. CTE33 improves the fit.

The agent loads the microstructure from Stage 1 automatically using the card ID.
You do not re-enter orientation or mass fraction.

### Example conversation

```
You:      INVERSE thermoelastic for card 5.
          CTE11 = 10.0 ppm/K
          CTE22 = 50.0 ppm/K
          CTE33 = 75.0 ppm/K

Agent:    Ready to run thermoelastic inverse:
            Card #5 (T300_PESU_CAMRI_Jan2026)
            Microstructure loaded from Stage 1.
            Free: f_cte1, f_cte2, m_cte
          Confirm?

You:      yes

Agent:    Thermoelastic inverse complete.

          Inferred constituent CTEs:
            f_cte1 (fiber longitudinal) = 2.50e-7 /K
            f_cte2 (fiber transverse)   = 2.85e-5 /K
            m_cte  (matrix)             = 6.19e-5 /K

          Fit error: 0.0041  (excellent)

          Save to the same card?

You:      yes

Agent:    Saved to Card #5.
```

---

## Stage 3 - Thermal Conductivity Characterization

**What you need:** a CSV file with thermal conductivity measured at multiple
temperatures. K11 is required at a minimum. The temperature range should span
at least 50 deg C for reliable results.

Prepare the file in this format:

```
temperature_C,K11_WmK,K22_WmK,K33_WmK
25,0.52,0.35,0.35
50,0.55,0.37,0.37
75,0.58,0.39,0.39
100,0.61,0.41,0.41
```

K22 and K33 columns are optional but improve the fit. Temperature must be in Celsius.

### Example conversation

```
You:      INVERSE thermal for card 5, CSV at /Users/me/Desktop/k_vs_T.csv

Agent:    Ready to run thermal inverse:
            Card #5 (T300_PESU_CAMRI_Jan2026)
            CSV: /Users/me/Desktop/k_vs_T.csv
            Channels detected: K11, K22, K33
            Temperature range: 25 deg C to 100 deg C
          Confirm?

You:      yes

Agent:    Thermal inverse complete.

          Inferred conductivities:
            k_f1 (fiber longitudinal) = 19.95 W/m.K
            k_f2 (fiber transverse)   =  3.38 W/m.K
            k_m  (matrix model):
              p1 = 0.00725 W/m.K
              p2 = 0.150   W/m.K

          Fit error: 0.0089  (excellent)

          Save to the same card?

You:      yes

Agent:    Saved to Card #5.
```

---

## Quick Characterization from a Spreadsheet

If you have all your measurements ready, put them in one CSV file. The agent
runs all applicable stages in sequence and saves the card automatically.

**Prepare the file:**

```
field,value,sigma
fiber,T300 Carbon Fiber (Toray),
polymer,PESU Ultrason (BASF),
printer,CAMRI,
card_name,T300_PESU_CAMRI_Jan2026,
E1_GPa,15.14,1.25
E2_GPa,5.14,0.30
E3_GPa,4.14,0.20
G12_GPa,2.50,0.15
nu12,0.35,
CTE11_ppm_K,10.0,
CTE22_ppm_K,50.0,
CTE33_ppm_K,75.0,
```

The `sigma` column is optional - leave it blank if you have no uncertainty values.
Use exact material and printer names from the database. Units: GPa for stiffness,
ppm/K for thermal expansion.

### Example conversation

```
You:      INVERSE characterize from /Users/me/Desktop/measurements.csv

Agent:    Read measurements.csv:
            Fiber:   T300 Carbon Fiber (Toray)
            Polymer: PESU Ultrason (BASF)
            Printer: CAMRI
            Card name: T300_PESU_CAMRI_Jan2026

            Elastic measurements: E1, E2, E3, G12, nu12
            Thermoelastic measurements: CTE11, CTE22, CTE33

          Running elastic inverse... done.  Fit error: 0.0031
          Running thermoelastic inverse... done.  Fit error: 0.0041

          Saved as Card #5 - T300_PESU_CAMRI_Jan2026
```

---

## Forward Prediction

Once a card is characterized, use it to predict properties - for example, to
check what the same materials would give on a different printer, or at a
different fiber content.

### Example conversations

```
You:      PREDICT what are the elastic properties for card 5?

Agent:    Predicted elastic properties for Card #5
          (T300_PESU_CAMRI_Jan2026):

            E1   = 15,514 MPa    E2   = 5,024 MPa    E3   = 4,240 MPa
            G12  =  2,519 MPa    G13  = 1,745 MPa    G23  = 1,556 MPa
            nu12 = 0.417         nu13 = 0.331         nu23 = 0.445
```

```
You:      PREDICT T300/PESU on a new printer with a11=0.60, a22=0.20,
          mf=0.25, ar=18

Agent:    Predicted elastic properties:

            E1   = 13,105 MPa    E2   = 6,210 MPa    E3   = 5,080 MPa
            G12  =  2,891 MPa    ...
```

---

## What-If Analysis

Ask the agent to sweep a parameter over a range to find a target property value
or to understand sensitivity.

### Example conversations

```
You:      PREDICT how much should I increase fiber mass fraction to get
          E1 = 20 GPa for card 5?

Agent:    Sweep: fiber_massfrac -> E1  (Card #5)

             Value          E1 (MPa)
          --------    --------------
            0.200        15,273.53
            0.250        19,141.86
            0.300        23,292.73

          At mf = 0.260, E1 ~ 19,925 MPa.
          Increase fiber mass fraction to approximately 0.261 to reach 20 GPa.
```

```
You:      PREDICT sweep ar from 10 to 30 for card 5, show E1

Agent:    Sweep: ar -> E1  (Card #5)

             Value          E1 (MPa)
          --------    --------------
            10.000         9,842.10
            15.000        12,910.45
            20.000        15,034.22
            25.000        16,289.33
            30.000        17,013.48
```

---

## Material and Theory Lookups

Use `SEARCH` for any question about material properties or composite mechanics.
The agent checks the knowledge base rather than guessing.

### Example conversations

```
You:      SEARCH what is the modulus of T300 carbon fiber?

Agent:    T300 Carbon Fiber (Toray):
            E1  = 230,000 MPa (longitudinal)
            E2  =  14,000 MPa (transverse)
            G12 =  28,000 MPa
            nu12 = 0.20
            rho  = 1,760 kg/m3
            (Source: fibers table, provenance: web)
```

```
You:      SEARCH what measurements do I need to identify fiber CTE?

Agent:    To reliably identify f_cte1 and f_cte2, you need CTE11 and CTE22
          from dilatometry. CTE33 is optional but improves the fit.
          The microstructure must already be known from an elastic inverse
          (Stage 1) before running the thermoelastic stage.
```

```
You:      SEARCH why is matrix Poisson ratio hard to identify from E1 and E2?

Agent:    E1 and E2 are dominated by fiber volume fraction and orientation.
          The matrix Poisson ratio affects shear coupling terms (G12, nu12)
          much more than the axial moduli. Without at least one shear or
          Poisson measurement in your target set, the Fisher Information
          for matrix_poisson is near zero - meaning many values give
          essentially the same E1/E2. The agent will fix it to the
          polymer datasheet value in this case.
```

---

## Troubleshooting

**"Material not found"**
The agent did not find your fiber or polymer name. Ask:
```
what materials do we have?
```
Use the exact name from that list, including any parentheses.

**Fit error above 0.10**
Your measurements may be inconsistent with each other, or there are not enough
of them to constrain all unknowns. Try:
- Adding G12 or nu12 if you only provided E1, E2, E3
- Checking units - the agent expects GPa for moduli, ppm/K for CTE
- Asking: *"why is my fit error so high?"*

**"Stage 1 must be saved first"**
You tried to run Stage 2 or Stage 3 before Stage 1 was complete and saved.
Finish Stage 1 and note the card ID, then retry with that ID.

**The agent asks for a specific number instead of accepting a range**
If you say *"ar is around 15 to 20"*, the agent will stop and ask you to
choose one value. Give your best estimate or the midpoint.

**I forgot my card ID**
```
what cards do we have?
```

---

## Unit Reference

| Quantity | Unit to use | Example |
|---|---|---|
| Stiffness (E, G) | GPa | `E1 = 15.14 GPa` |
| Poisson ratio | dimensionless | `nu12 = 0.35` |
| Thermal expansion (CTE) | ppm/K | `CTE11 = 10.0 ppm/K` |
| Thermal conductivity (K) | W/m.K | provided in the CSV |
| Temperature | Celsius | `25, 50, 75, 100` |
