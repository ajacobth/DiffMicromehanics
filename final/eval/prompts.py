"""
prompts.py — Benchmark data for MateriAl agent evaluation.

Contains:
  PROMPTS          — 16 natural-language prompts (E, T, K, S categories)
  GUI_INPUTS       — exact values to enter in gui.py for each prompt
  SCORED_PROPERTIES — which output fields to extract and compare per prompt
  EXPECTED_TOOL    — which tool the agent should call for each prompt
"""

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {

    # Elastic — tests input parsing, unit handling, orientation variety
    "E1": (
        "Predict the elastic properties of a short carbon fiber polmyer composite "
        "with the following properties: fiber axial modulus 230 GPa, fiber transverse "
        "modulus 15 GPa, fiber shear modulus 15 GPa, fiber axial Poisson 0.20, fiber "
        "transverse Poisson 0.25, matrix modulus 1600 MPa, matrix Poisson 0.40, fiber "
        "density 1760 kg/m3, matrix density 1010 kg/m3, fiber mass fraction 0.20, "
        "aspect ratio 20, orientation tensor a11=0.80, a22=0.10, a12=0.02, a13=0.01, a23=0.01."
    ),
    "E2": (
        "I have a carbon fiber ABS system. The fiber is moderately aligned with "
        "a11=0.65, a22=0.20, a12=0.03, a13=0.02, a23=0.02. Fiber modulus is 230 GPa "
        "axially and 15 GPa transversely, shear modulus 15 GPa, nu_f12=0.20, nu_f23=0.25. "
        "Matrix modulus is 3200 MPa, Poisson 0.33, densities 1760 and 1250 kg/m3. "
        "Mass fraction is 20 percent, aspect ratio 20. What are the composite elastic constants?"
    ),
    "E3": (
        "Predict E1, E2, E3, G12, G13, G23, nu12, nu13, nu23 for short CF/PA12 with "
        "near-planar isotropic orientation: a11=0.45, a22=0.40, a12=0.04, a13=0.01, a23=0.01. "
        "Use fiber E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25. "
        "Matrix E_m=1600 MPa, nu_m=0.30. Densities: fiber 1760, matrix 1250 kg/m3. "
        "Mass fraction 0.30, AR=20."
    ),
    "E4": (
        "What are E1 and E2 for a highly aligned short fiber carbon PESU system with "
        "a11=0.85, a22=0.08, a12=0.01, a13=0.01, a23=0.01? Fiber: E_f1=230 GPa, "
        "E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25. Matrix: E_m=1600 MPa, "
        "nu_m=0.35. Fiber density 1760, matrix density 1350 kg/m3. Mass fraction 0.25, AR=25."
    ),
    "E5": (
        "Predict all nine elastic constants for a short fiber composite. Fiber: axial "
        "stiffness 230000 MPa, transverse stiffness 15000 MPa, shear 15000 MPa, Poisson "
        "ratios 0.20 and 0.25. Matrix: 1.6 GPa stiffness, Poisson 0.33. Densities 1760 "
        "and 1010. Fiber content 30 percent by mass, aspect ratio 20, orientation "
        "a11=0.70, a22=0.15, a12=0.02, a13=0.01, a23=0.01."
        
        
    ),
    # ambiguous orientation tensor
    "E6": (
        "I have a carbon fiber ABS system. The fibers are randomly aligned. Fiber modulus is 230 GPa "
        "axially and 15 GPa transversely, shear modulus 15 GPa, nu_f12=0.20, nu_f23=0.25. "
        "Matrix modulus is 1600 MPa, Poisson 0.40, densities 1760 and 1010 kg/m3. "
        "Mass fraction is 30 percent, aspect ratio 20. What are the composite elastic constants?"
        
    ),

    # Thermoelastic — CTE extraction + unit conversion
    "T1": (
        "Predict the coefficients of thermal expansion for CF PESU system with orientation "
        "a11=0.70, a22=0.10, a12=0.02, a13=0.01, a23=0.01. Fiber axial CTE is -0.5 ppm/K, "
        "fiber transverse CTE is 12 ppm/K, matrix CTE is 60 ppm/K. Elastic inputs: "
        "E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, E_m=1600 MPa, "
        "nu_m=0.33, densities 1760 and 1230 kg/m3, mass fraction 0.30, AR=20."
    ),
    "T2": (
        "What are the composite CTEs for a moderately aligned short carbon fiber PEI system? "
        "Orientation: a11=0.65, a22=0.20, a12=0.03, a13=0.02, a23=0.02. Fiber CTE "
        "longitudinal -0.5 ppm/K, transverse 12 ppm/K. Matrix CTE 80 ppm/K. Elastic "
        "properties: E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, "
        "E_m=1600 MPa, nu_m=0.33, densities 1760 and 1200 kg/m3, mass fraction 25 percent, AR=20."
    ),
    "T3": (
        "Predict alpha11, alpha22, and alpha33 for a near-planar isotropic CF/PESU system "
        "with a11=0.45, a22=0.40, a12=0.04, a13=0.01, a23=0.01. Fiber CTEs: "
        "f_cte1=-0.5 ppm/K, f_cte2=12 ppm/K. Matrix CTE: 80 ppm/K. Fiber: E_f1=230 GPa, "
        "E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25. Matrix: E_m=1600 MPa, "
        "nu_m=0.35. Densities 1760 and 1350 kg/m3, mass fraction 0.25, AR=20."
    ),
    "T4": (
        # Unit conversion test: CTE given in 1/K not ppm/K
        "What is the CTE tensor for CF/PA12 with a11=0.75, a22=0.12, a12=0.02, a13=0.01, "
        "a23=0.01? Fiber longitudinal CTE: -5e-7 per K. Fiber transverse CTE: 1.2e-5 per K. "
        "Matrix CTE: 8e-5 per K. Elastic inputs: fiber 230 GPa axial, 15 GPa transverse, "
        "15 GPa shear, Poisson 0.20 and 0.25. Matrix 1600 MPa, Poisson 0.35. "
        "Densities 1760 and 1330 kg/m3. Mass fraction 0.25, AR=20."
    ),

    # Thermal conductivity
    "K1": (
        "Predict the thermal conductivity tensor between 25 and 150 degrees C for CF/PESU "
        "with a11=0.80, a22=0.10, a12=0.02, a13=0.01, a23=0.01. Fiber conductivities: "
        "k_f1=8 W/mK axial, k_f2=1 W/mK transverse. Matrix conductivity model: "
        "k_m(T) = 0.02*sqrt(T) + 0.18 W/mK. Mass fraction 0.20, AR=20, densities "
        "1760 and 1200 kg/m3."
    ),
    "K2": (
        "What is K11, K22, K33 for CF/PA12 at 50 and 100 degrees C? Orientation a11=0.65, "
        "a22=0.20, a12=0.03, a13=0.02, a23=0.02. Fiber k_f1=8 W/mK, k_f2=1 W/mK. "
        "Matrix: p1=0.02, p2=0.18 in the square root model. Mass fraction 0.30, AR=20, "
        "densities 1760 and 1250 kg/m3."
    ),
    "K3": (
        "How thermally conductive is a short fiber carbon nylon part printed with highly "
        "aligned fibers? Use a11=0.80, a22=0.10, a12=0.02, a13=0.01, a23=0.01. Fiber has "
        "longitudinal conductivity 8 W/mK and transverse 1 W/mK. The nylon matrix follows "
        "k_m = 0.02*sqrt(T) + 0.18. Fiber mass fraction 30 percent, aspect ratio 20, "
        "densities 1760 and 1210 kg/m3. Give me values at 25, 50, 75, and 100 degrees C."
    ),

    # Parameter sweeps
    "S1": (
        "How does E1 change as fiber mass fraction increases from 0.10 to 0.25? "
        "Orientation: a11=0.80, a22=0.10, a12=0.02, a13=0.01, a23=0.01. "
        "Fiber: E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, density 1760 kg/m3. "
        "Matrix: E_m=1600 MPa, nu_m=0.40, density 1250 kg/m3. AR=20."
    ),
    "S2": (
        "Show me how composite E1 varies with fiber alignment as a11 goes from 0.50 to 0.90. "
        "Keep a22=0.10, a12=0.02, a13=0.01, a23=0.01, mass fraction 0.20, AR=18. "
        "Fiber: E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, density 1760 kg/m3. "
        "Matrix: E_m=3200 MPa, nu_m=0.33, density 1300 kg/m3."
    ),
    "S3": (
        "How does G12 change as fiber aspect ratio varies from 5 to 30? "
        "Orientation: a11=0.70, a22=0.15, a12=0.02, a13=0.01, a23=0.01. "
        "Fiber: E_f1=230 GPa, E_f2=15 GPa, G_f12=15 GPa, nu_f12=0.20, nu_f23=0.25, density 1760 kg/m3. "
        "Matrix: E_m=1600 MPa, nu_m=0.33, density 1330 kg/m3. Mass fraction 0.30."
    ),
}


# ── GUI input worksheet ───────────────────────────────────────────────────────
# Values to enter manually in gui.py to 

GUI_INPUTS: dict[str, dict] = {
    # ── Elastic ────────────────────────────────────────────────────────────────
    "E1": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.40  ρ=1010 kg/m³",
        "micro":  "a11=0.80  a22=0.10  a12=0.02  a13=0.01  a23=0.01  mf=0.20  AR=20",
    },
    "E2": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=3200 MPa  ν=0.33  ρ=1250 kg/m³",
        "micro":  "a11=0.65  a22=0.20  a12=0.03  a13=0.02  a23=0.02  mf=0.20  AR=20",
        "note":   "ABS matrix: 3200 MPa, ν=0.33, ρ=1250 kg/m³, mf=20%",
    },
    "E3": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.30  ρ=1250 kg/m³",
        "micro":  "a11=0.45  a22=0.40  a12=0.04  a13=0.01  a23=0.01  mf=0.30  AR=20",
        "note":   "PA12 matrix: ν=0.30, ρ=1250 kg/m³",
    },
    "E4": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.35  ρ=1350 kg/m³",
        "micro":  "a11=0.85  a22=0.08  a12=0.01  a13=0.01  a23=0.01  mf=0.25  AR=25",
        "note":   "PESU matrix: ν=0.35, ρ=1350 kg/m³",
    },
    "E5": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.33  ρ=1010 kg/m³",
        "micro":  "a11=0.70  a22=0.15  a12=0.02  a13=0.01  a23=0.01  mf=0.30  AR=20",
        "note":   "Tests mixed units in prompt (MPa/GPa) — matrix ν=0.33",
    },
    "E6": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.40  ρ=1010 kg/m³",
        "micro":  "a11=0.33  a22=0.33  a12=0.00  a13=0.00  a23=0.00  mf=0.30  AR=20",
        "note":   "Tests 'randomly aligned' keyword → a11=0.33 a22=0.33 a33=0.34",
    },

    # ── Thermoelastic ──────────────────────────────────────────────────────────
    "T1": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.33  ρ=1230 kg/m³",
        "micro":  "a11=0.70  a22=0.10  a12=0.02  a13=0.01  a23=0.01  mf=0.30  AR=20",
        "CTE":    "f_cte1=-0.5e-6 /K   f_cte2=12e-6 /K   m_cte=60e-6 /K",
        "note":   "CF/PESU: ν=0.33, ρ=1230 kg/m³, a11=0.70, m_cte=60 ppm/K",
    },
    "T2": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.33  ρ=1200 kg/m³",
        "micro":  "a11=0.65  a22=0.20  a12=0.03  a13=0.02  a23=0.02  mf=0.25  AR=20",
        "CTE":    "f_cte1=-0.5e-6 /K   f_cte2=12e-6 /K   m_cte=80e-6 /K",
        "note":   "CF/PEI: ν=0.33, ρ=1200 kg/m³, mf=25%",
    },
    "T3": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.35  ρ=1350 kg/m³",
        "micro":  "a11=0.45  a22=0.40  a12=0.04  a13=0.01  a23=0.01  mf=0.25  AR=20",
        "CTE":    "f_cte1=-0.5e-6 /K   f_cte2=12e-6 /K   m_cte=80e-6 /K",
        "note":   "CF/PESU near-planar: ν=0.35, ρ=1350 kg/m³, mf=25%",
    },
    "T4": {
        "fiber":  "E1=230 GPa  E2=15 GPa  G12=15 GPa  ν12=0.20  ν23=0.25  ρ=1760 kg/m³",
        "matrix": "E=1600 MPa  ν=0.35  ρ=1330 kg/m³",
        "micro":  "a11=0.75  a22=0.12  a12=0.02  a13=0.01  a23=0.01  mf=0.25  AR=20",
        "CTE":    "f_cte1=-5e-7 /K   f_cte2=1.2e-5 /K   m_cte=8e-5 /K",
        "note":   "Tests 1/K input (not ppm/K). CF/PA12: ν=0.35, ρ=1330 kg/m³, mf=25%",
    },

    # ── Thermal conductivity ───────────────────────────────────────────────────
    "K1": {
        "micro":   "a11=0.80  a22=0.10  a12=0.02  a13=0.01  a23=0.01  mf=0.20  AR=20",
        "density": "fiber=1760 kg/m³  matrix=1200 kg/m³",
        "k":       "k_f1=8.0 W/mK  k_f2=1.0 W/mK  k_m(T)=0.02·√T + 0.18 W/mK",
        "score_at":"25°C",
        "note":    "CF/PESU: mf=20%, ρ_m=1200 kg/m³",
    },
    "K2": {
        "micro":   "a11=0.65  a22=0.20  a12=0.03  a13=0.02  a23=0.02  mf=0.30  AR=20",
        "density": "fiber=1760 kg/m³  matrix=1250 kg/m³",
        "k":       "k_f1=8.0 W/mK  k_f2=1.0 W/mK  k_m(T)=0.02·√T + 0.18 W/mK",
        "score_at":"50°C and 100°C",
        "note":    "CF/PA12: ρ_m=1250 kg/m³",
    },
    "K3": {
        "micro":   "a11=0.80  a22=0.10  a12=0.02  a13=0.01  a23=0.01  mf=0.30  AR=20",
        "density": "fiber=1760 kg/m³  matrix=1210 kg/m³",
        "k":       "k_f1=8.0 W/mK  k_f2=1.0 W/mK  k_m(T)=0.02·√T + 0.18 W/mK",
        "score_at":"25°C and 100°C",
        "note":    "CF/nylon: ρ_m=1210 kg/m³",
    },

    # ── Sweeps — tool routing only ─────────────────────────────────────────────
    "S1": {
        "note": "Sweep mf 0.10→0.25. CF/PESU: E_m=1600 MPa, ν=0.40, ρ_m=1250, AR=20. No GUI value to fill.",
    },
    "S2": {
        "note": "Sweep a11 0.50→0.90. CF/ABS: E_m=3200 MPa, ν=0.33, ρ_m=1300, mf=0.20, AR=18. No GUI value to fill.",
    },
    "S3": {
        "note": "Sweep AR 5→30. CF/PA12: E_m=1600 MPa, ν=0.33, ρ_m=1330, mf=0.30. No GUI value to fill.",
    },
}


# ── Which output properties to extract and score per prompt ───────────────────
# These must match what parse_tool_output() can extract from the tool text.
# Thermal properties: key format is  k{component}_{temp}C  e.g. k11_25C

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
    "T4": ["E1", "E2", "G12", "nu12", "CTE11", "CTE22", "CTE33"],
    "K1": ["k11_25C",  "k22_25C",  "k33_25C"],
    "K2": ["k11_50C",  "k22_50C",  "k33_50C",
           "k11_100C", "k22_100C", "k33_100C"],
    "K3": ["k11_25C",  "k22_25C",  "k33_25C",
           "k11_100C", "k22_100C", "k33_100C"],
    "S1": [],   # sweep: tool routing check only
    "S2": [],
    "S3": [],
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
    "T4": "predict_properties",
    "K1": "predict_thermal_conductivity",
    "K2": "predict_thermal_conductivity",
    "K3": "predict_thermal_conductivity",
    "S1": "sweep_parameter",
    "S2": "sweep_parameter",
    "S3": "sweep_parameter",
}
