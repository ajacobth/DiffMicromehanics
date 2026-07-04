"""
prompts.py — Benchmark data for MateriAl agent evaluation.

Contains:
  PROMPT_GOALS      — one-line test objective per prompt
  PROMPTS           — 12 natural-language prompts (E, T, K categories)
  GUI_INPUTS        — exact values to enter in gui.py for each prompt
  SCORED_PROPERTIES — which output fields to extract and compare per prompt
  EXPECTED_TOOL     — which tool the agent should call for each prompt

Prompt design principle
-----------------------
Each category introduces one new parsing / routing challenge per prompt.
Within a category the prompts roughly increase in ambiguity so early prompts
act as a sanity baseline and later ones stress-test edge cases.
"""

# ── Test objectives ───────────────────────────────────────────────────────────

PROMPT_GOALS: dict[str, str] = {
    # Elastic
    "E1": "Baseline: all inputs spelled out with explicit labels and GPa units.",
    "E2": "Material name ('CF ABS') alongside explicit numbers — agent must use Option C, not DB lookup.",
    "E3": "User lists desired outputs up front — agent still calls full tool and returns all nine constants.",
    "E4": "Partial output request ('E1 and E2') — agent should call full tool and return everything.",
    "E5": "Mixed units in same prompt (fiber in MPa, matrix in GPa) — unit normalisation required.",
    "E6": "Orientation keyword 'randomly aligned' — vocabulary lookup to a11=0.33 a22=0.33 required.",
    # Thermoelastic
    "T1": "Baseline: CTE in ppm/K, explicit orientation, standard material params.",
    "T2": "Generic 'carbon fiber polymer system' (no named polymer) with explicit numbers — agent must use Option C without any DB lookup.",
    "T3": "Contradictory orientation request: 'planar isotropic with a33=0.1' (planar iso implies a33≈0) — agent should flag inconsistency and resolve to a11=a22=0.45. CTE given in 1/K, not ppm/K.",
    # Thermal conductivity
    "K1": "Informal k_m label ('polymer conds is 1.2 W/mK') — agent must extract k_m from natural language and run successfully.",
    "K2": "Parametric k_m model (p1, p2 given) evaluated at two temperatures.",
    "K3": "Orientation keyword 'randomly oriented' + k_m as written formula — combines keyword resolution with parametric path.",
}


# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {

    # ── Elastic ────────────────────────────────────────────────────────────────
    # E1: baseline — every value labelled, GPa units throughout
    "E1": (
        "Predict the elastic properties of a short carbon fiber polmyer composite "
        "with the following properties: fiber axial modulus 230 GPa, fiber transverse "
        "modulus 15 GPa, fiber shear modulus 15 GPa, fiber axial Poisson 0.20, fiber "
        "transverse Poisson 0.25, matrix modulus 1600 MPa, matrix Poisson 0.40, fiber "
        "density 1760 kg/m3, matrix density 1010 kg/m3, fiber mass fraction 0.20, "
        "aspect ratio 20, orientation tensor a11=0.80, a22=0.10, a12=0.02, a13=0.01, a23=0.01."
    ),
    # E2: material name label alongside explicit numbers — must NOT trigger DB lookup
    "E2": (
        "I have a carbon fiber ABS system. The fiber is moderately aligned with "
        "a11=0.65, a22=0.20, a12=0.03, a13=0.02, a23=0.02. Fiber modulus is 230 GPa "
        "axially and 15 GPa transversely, shear modulus 15 GPa, nu_f12=0.20, nu_f23=0.25. "
        "Matrix modulus is 3200 MPa, Poisson 0.33, densities 1760 and 1250 kg/m3. "
        "Mass fraction is 20 percent, aspect ratio 20. What are the composite elastic constants?"
    ),
    # E3: user lists desired outputs — agent must still return all nine constants
    "E3": (
        "Predict E1, E2, E3, G12, G13, G23, nu12, nu13, nu23 for short CF/PA12 with "
        "near-planar isotropic orientation: a11=0.45, a22=0.40, a12=0.04, a13=0.01, a23=0.01. "
        "Use fiber E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25. "
        "Matrix E_m=1600 MPa, nu_m=0.30. Densities: fiber 1760, matrix 1250 kg/m3. "
        "Mass fraction 0.30, AR=20."
    ),
    # E4: partial output request — agent must still call full tool
    "E4": (
        "What are E1 and E2 for a highly aligned short fiber carbon PESU system with "
        "a11=0.85, a22=0.08, a12=0.01, a13=0.01, a23=0.01? Fiber: E_f1=230 GPa, "
        "E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25. Matrix: E_m=1600 MPa, "
        "nu_m=0.35. Fiber density 1760, matrix density 1350 kg/m3. Mass fraction 0.25, AR=25."
    ),
    # E5: mixed units (fiber in MPa, matrix in GPa) — unit normalisation stress test
    "E5": (
        "Predict all nine elastic constants for a short fiber composite. Fiber: axial "
        "stiffness 230000 MPa, transverse stiffness 15000 MPa, shear 15000 MPa, Poisson "
        "ratios 0.20 and 0.25. Matrix: 1.6 GPa stiffness, Poisson 0.33. Densities 1760 "
        "and 1010. Fiber content 30 percent by mass, aspect ratio 20, orientation "
        "a11=0.70, a22=0.15, a12=0.02, a13=0.01, a23=0.01."
    ),
    # E6: orientation keyword — vocabulary lookup required (randomly aligned → a11=0.33 a22=0.33)
    "E6": (
        "I have a carbon fiber ABS system. The fibers are randomly aligned. Fiber modulus is 230 GPa "
        "axially and 15 GPa transversely, shear modulus 15 GPa, nu_f12=0.20, nu_f23=0.25. "
        "Matrix modulus is 1600 MPa, Poisson 0.40, densities 1760 and 1010 kg/m3. "
        "Mass fraction is 30 percent, aspect ratio 20. What are the composite elastic constants?"
    ),

    # ── Thermoelastic ──────────────────────────────────────────────────────────
    # T1: baseline — CTE in ppm/K, standard orientation
    "T1": (
        "Predict the coefficients of thermal expansion for CF PESU system with orientation "
        "a11=0.70, a22=0.10, a12=0.02, a13=0.01, a23=0.01. Fiber axial CTE is -0.5 ppm/K, "
        "fiber transverse CTE is 12 ppm/K, matrix CTE is 60 ppm/K. Elastic inputs: "
        "E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, E_m=1600 MPa, "
        "nu_m=0.33, densities 1760 and 1230 kg/m3, mass fraction 0.30, AR=20."
    ),
    # T2: higher matrix CTE (80 ppm/K), mf=25%
    "T2": (
        "What are the composite CTEs for a moderately aligned short carbon fiber polymer system? "
        "Orientation: a11=0.65, a22=0.20, a12=0.03, a13=0.02, a23=0.02. Fiber CTE "
        "longitudinal -0.5 ppm/K, transverse 12 ppm/K. Matrix CTE 80 ppm/K. Elastic "
        "properties: E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, "
        "E_m=1600 MPa, nu_m=0.33, densities 1760 and 1200 kg/m3, mass fraction 25 percent, AR=20."
    ),

    # T3: Very challenging case since planar isotrpoy cannot have a33 to be 0.1
    "T3": (
        "What is the CTE tensor for CF/PA12 with planar isotropic orientation with a33 to be 0.1, "
        "Fiber longitudinal CTE: -5e-7 per K. Fiber transverse CTE: 1.2e-5 per K. "
        "Matrix CTE: 8e-5 per K. Elastic inputs: fiber 230 GPa axial, 15 GPa transverse, "
        "15 GPa shear, Poisson 0.20 and 0.25. Matrix 1600 MPa, Poisson 0.35. "
        "Densities 1760 and 1330 kg/m3. Mass fraction 0.25, AR=20."
    ),

    # ── Thermal conductivity ───────────────────────────────────────────────────
    # K1: intentionally incomplete — k_m omitted; expected agent behaviour: CANNOT RUN
    "K1": (
        "Predict the thermal conductivity tensor for CF/PESU "
        "with a11=0.80, a22=0.10, a12=0.02, a13=0.01, a23=0.01. Fiber conductivities: "
        "8 W/mK axial, and 1 W/mK transverse polymer conds is 1.2W/mK Mass fraction 0.20, AR=20, densities "
        "1760 and 1200 kg/m3."
    ),
    # K2: parametric k_m model at two temperatures
    "K2": (
        "What is K11, K22, K33 for CF/PA12 at 50 and 100 degrees C? Orientation a11=0.65, "
        "a22=0.20, a12=0.03, a13=0.02, a23=0.02. Fiber k_f1=8 W/mK, k_f2=1 W/mK. "
        "Matrix: p1=0.02, p2=0.18 in the square root model. Mass fraction 0.30, AR=20, "
        "densities 1760 and 1250 kg/m3."
    ),
    # K3: orientation keyword + k_m as written formula
    "K3": (
        "How thermally conductive is a short fiber carbon nylon part printed with randomly oriented fibers."
        " Fiber has longitudinal conductivity 8 W/mK and transverse 1 W/mK. The nylon matrix follows "
        "k_m = 0.02*sqrt(T) + 0.18. Fiber mass fraction 30 percent, aspect ratio 20, "
        "densities 1760 and 1210 kg/m3. Give me values at 25, 50, 75, and 100 degrees C."
    ),

    # ── Parameter sweeps ───────────────────────────────────────────────────────
    # S1–S3: tool routing only — no numerical scoring
    #"S1": (
    #    "How does E1 change as fiber mass fraction increases from 0.10 to 0.25? "
    #    "Orientation: a11=0.80, a22=0.10, a12=0.02, a13=0.01, a23=0.01. "
    #    "Fiber: E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, density 1760 kg/m3. "
    #    "Matrix: E_m=1600 MPa, nu_m=0.40, density 1250 kg/m3. AR=20."
    #),
    #"S2": (
    #    "Show me how composite E1 varies with fiber alignment as a11 goes from 0.50 to 0.90. "
    #    "Keep a22=0.10, a12=0.02, a13=0.01, a23=0.01, mass fraction 0.20, AR=18. "
    #    "Fiber: E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, density 1760 kg/m3. "
    #    "Matrix: E_m=3200 MPa, nu_m=0.33, density 1300 kg/m3."
    #),
    #"S3": (
    #    "How does G12 change as fiber aspect ratio varies from 5 to 30? "
    #    "Orientation: a11=0.70, a22=0.15, a12=0.02, a13=0.01, a23=0.01. "
    #    "Fiber: E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, density 1760 kg/m3. "
    #    "Matrix: E_m=1600 MPa, nu_m=0.33, density 1330 kg/m3. Mass fraction 0.30."
    #),
}


# ── GUI input worksheet ───────────────────────────────────────────────────────
# Exact values to enter in gui.py to produce the ground-truth reference output.
# Fill the gui_value column in the CSV with the numbers gui.py reports, then
# run --score to compute MAPE.

GUI_INPUTS: dict[str, dict] = {
    # ── Elastic ────────────────────────────────────────────────────────────────
    "E1": {
        "goal":   PROMPT_GOALS["E1"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.40  ρ=1010 kg/m³",
        "micro":  "a11=0.80  a22=0.10  a12=0.02  a13=0.01  a23=0.01  mf=0.20  AR=20",
    },
    "E2": {
        "goal":   PROMPT_GOALS["E2"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=3200 MPa  ν=0.33  ρ=1250 kg/m³",
        "micro":  "a11=0.65  a22=0.20  a12=0.03  a13=0.02  a23=0.02  mf=0.20  AR=20",
        "note":   "ABS matrix: 3200 MPa, ν=0.33, ρ=1250 kg/m³. Agent must NOT call get_material_details.",
    },
    "E3": {
        "goal":   PROMPT_GOALS["E3"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.30  ρ=1250 kg/m³",
        "micro":  "a11=0.45  a22=0.40  a12=0.04  a13=0.01  a23=0.01  mf=0.30  AR=20",
        "note":   "PA12 matrix: ν=0.30, ρ=1250 kg/m³. Agent should return all nine constants.",
    },
    "E4": {
        "goal":   PROMPT_GOALS["E4"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.35  ρ=1350 kg/m³",
        "micro":  "a11=0.85  a22=0.08  a12=0.01  a13=0.01  a23=0.01  mf=0.25  AR=25",
        "note":   "PESU matrix: ν=0.35, ρ=1350 kg/m³. Agent asked only for E1/E2 but must call full tool.",
    },
    "E5": {
        "goal":   PROMPT_GOALS["E5"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.33  ρ=1010 kg/m³",
        "micro":  "a11=0.70  a22=0.15  a12=0.02  a13=0.01  a23=0.01  mf=0.30  AR=20",
        "note":   "Fiber given in MPa, matrix in GPa — same reference values once converted.",
    },
    "E6": {
        "goal":   PROMPT_GOALS["E6"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.40  ρ=1010 kg/m³",
        "micro":  "a11=0.33  a22=0.33  a12=0.00  a13=0.00  a23=0.00  mf=0.30  AR=20",
        "note":   "'randomly aligned' → 3D isotropic: a11=0.33, a22=0.33, a33=0.34.",
    },

    # ── Thermoelastic ──────────────────────────────────────────────────────────
    "T1": {
        "goal":   PROMPT_GOALS["T1"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.33  ρ=1230 kg/m³",
        "micro":  "a11=0.70  a22=0.10  a12=0.02  a13=0.01  a23=0.01  mf=0.30  AR=20",
        "CTE":    "f_cte1=-0.5e-6 /K   f_cte2=12e-6 /K   m_cte=60e-6 /K",
        "note":   "CF/PESU: ν=0.33, ρ=1230 kg/m³, m_cte=60 ppm/K.",
    },
    "T2": {
        "goal":   PROMPT_GOALS["T2"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.33  ρ=1200 kg/m³",
        "micro":  "a11=0.65  a22=0.20  a12=0.03  a13=0.02  a23=0.02  mf=0.25  AR=20",
        "CTE":    "f_cte1=-0.5e-6 /K   f_cte2=12e-6 /K   m_cte=80e-6 /K",
        "note":   "No named polymer — just 'polymer system'. All values explicit so Option C. ν=0.33, ρ=1200 kg/m³, mf=25%.",
    },
    "T3": {
        "goal":   PROMPT_GOALS["T3"],
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.35  ρ=1330 kg/m³",
        "micro":  "a11=0.45  a22=0.45  a12=0.00  a13=0.00  a23=0.00  mf=0.25  AR=20",
        "CTE":    "f_cte1=-5e-7 /K   f_cte2=1.2e-5 /K   m_cte=8e-5 /K",
        "note":   (
            "Planar isotropic + a33=0.1 is contradictory (planar iso → a33≈0). "
            "Correct resolution: a11=a22=(1−0.1)/2=0.45. "
            "CTE in 1/K notation — agent must pass as-is, no ×1e-6. "
            "CF/PA12: ν=0.35, ρ=1330 kg/m³, mf=25%."
        ),
    },

    # ── Thermal conductivity ───────────────────────────────────────────────────
    "K1": {
        "goal":    PROMPT_GOALS["K1"],
        "micro":   "a11=0.80  a22=0.10  a12=0.02  a13=0.01  a23=0.01  mf=0.20  AR=20",
        "density": "fiber=1760 kg/m³  matrix=1200 kg/m³",
        "k":       "k_f1=8.0 W/mK  k_f2=1.0 W/mK  k_m=1.2 W/mK (scalar)",
        "score_at":"25°C  (scalar k_m, no temperature dependence)",
        "note":    "k_m given informally as 'polymer conds is 1.2 W/mK' — agent must extract k_m=1.2 and call the tool.",
    },
    "K2": {
        "goal":    PROMPT_GOALS["K2"],
        "micro":   "a11=0.65  a22=0.20  a12=0.03  a13=0.02  a23=0.02  mf=0.30  AR=20",
        "density": "fiber=1760 kg/m³  matrix=1250 kg/m³",
        "k":       "k_f1=8.0 W/mK  k_f2=1.0 W/mK  k_m(T)=0.02·√T + 0.18 W/mK",
        "score_at":"50°C and 100°C",
        "note":    "CF/PA12: ρ_m=1250 kg/m³. Parametric path via p1=0.02, p2=0.18.",
    },
    "K3": {
        "goal":    PROMPT_GOALS["K3"],
        "micro":   "a11=0.33  a22=0.33  a12=0.00  a13=0.00  a23=0.00  mf=0.30  AR=20",
        "density": "fiber=1760 kg/m³  matrix=1210 kg/m³",
        "k":       "k_f1=8.0 W/mK  k_f2=1.0 W/mK  k_m(T)=0.02·√T + 0.18 W/mK",
        "score_at":"25°C and 100°C",
        "note":    (
            "'randomly oriented' → 3D isotropic: a11=0.33, a22=0.33, a33=0.34. "
            "CF/nylon: ρ_m=1210 kg/m³. k_m given as formula — agent must extract p1=0.02, p2=0.18."
        ),
    },

}


# ── Which output properties to extract and score per prompt ───────────────────
# Key format for thermal: k{component}_{temp}C  e.g. k11_25C
# K1 is excluded from numerical scoring (CANNOT RUN — k_m missing from prompt).

SCORED_PROPERTIES: dict[str, list[str]] = {
    "E1": ["E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23"],
    "E2": ["E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23"],
    "E3": ["E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23"],
    "E4": ["E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23"],
    "E5": ["E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23"],
    "E6": ["E1", "E2", "E3", "G12", "G13", "G23", "nu12", "nu13", "nu23"],
    "T1": ["E1", "E2", "G12", "nu12", "CTE11", "CTE22", "CTE33"],
    "T2": ["E1", "E2", "G12", "nu12", "CTE11", "CTE22", "CTE33"],
    "T3": ["E1", "E2", "G12", "nu12", "CTE11", "CTE22", "CTE33"],
    "K1": ["k11_25C", "k22_25C", "k33_25C"],   # scalar k_m=1.2 at 25°C
    "K2": ["k11_50C",  "k22_50C",  "k33_50C",
           "k11_100C", "k22_100C", "k33_100C"],
    "K3": ["k11_25C",  "k22_25C",  "k33_25C",
           "k11_100C", "k22_100C", "k33_100C"],
}


# ── Expected tool per prompt ──────────────────────────────────────────────────

EXPECTED_TOOL: dict[str, str] = {
    "E1": "predict_properties",
    "E2": "predict_properties",
    "E3": "predict_properties",
    "E4": "predict_properties",
    "E5": "predict_properties",
    "E6": "predict_properties",
    "T1": "predict_properties",
    "T2": "predict_properties",
    "T3": "predict_properties",
    "K1": "predict_thermal_conductivity",
    "K2": "predict_thermal_conductivity",
    "K3": "predict_thermal_conductivity",
}
