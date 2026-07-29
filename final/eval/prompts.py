"""
prompts.py — Benchmark data for MateriAl agent evaluation.

Contains:
  PROMPT_GOALS      — one-line test objective per prompt
  PROMPTS           — natural-language prompts (E, T, K, IE, IT categories)
  GUI_INPUTS        — exact values to enter in gui.py for each forward prompt
  SCORED_PROPERTIES — which output fields to extract and compare per prompt
  EXPECTED_TOOL     — which tool the agent should call for each prompt

Prompt design principle
-----------------------
Each category introduces one new parsing / routing challenge per prompt.
Within a category the prompts roughly increase in ambiguity so early prompts
act as a sanity baseline and later ones stress-test edge cases.

Inverse prompt categories (IE/IT):
  IE — elastic inverse (run_elastic_inverse).  Self-contained: no prior card needed.
       Scored on fit_error from solver output (< 0.05 = converged = inputs were correct).
       Known-answer: synthetic measurements from forward run with ground-truth inputs:
         T300/PESU/CAMRI,  E_m=3100 MPa,  a11=0.70 a22=0.15 mf=0.25 ar=30
         Predicted: E1=20654 E2=6338 E3=6257 G12=3287 nu12=0.40 nu13=0.40 (all MPa)

  IT — thermoelastic inverse (run_thermoelastic_inverse).
       Requires a saved elastic inverse card (card_id=1 assumed).
       Synthetic CTE targets from same ground truth:
         CTE11=5.26 ppm/K  CTE22=45.05 ppm/K  CTE33=44.50 ppm/K

  Multi-turn: inverse prompts require a confirmation step. The eval runner sends
  a follow-up "yes, go ahead" if the expected tool was not called on the first turn.
"""

# ── Test objectives ───────────────────────────────────────────────────────────

PROMPT_GOALS: dict[str, str] = {
    # ── Inverse: elastic ──────────────────────────────────────────────────────
    "IE1": "Baseline: E1+E2+G12+nu12 in GPa, named DB materials, go-ahead in prompt.",
    "IE2": "Measurements given in MPa (unusual) — agent must still pass them correctly.",
    # IE3: identifiability multi-turn flow — run manually, not in automated eval
    "IE3": "CT-measured orientation (a11/a22 fixed) combined with elastic measurements.",
    "IE4": "Ambiguous range input ('~20 GPa') — RULE 1 should block tool call.",

    # ── Inverse: thermoelastic ────────────────────────────────────────────────
    "IT1": "Baseline: CTE in ppm/K, card_id provided explicitly.",
    "IT2": "CTE in 1/K scientific notation — no unit conversion needed but must parse.",

    # ── Inverse: thermal ─────────────────────────────────────────────────────
    "IK1": "All 3 K channels, wide T range (25–200°C) — explicit Stage 1 inputs, should converge cleanly.",
    "IK2": "K11 only, narrow T range (25–50°C) — tool should warn that missing K22/K33 may hurt accuracy.",

    # ── Forward ───────────────────────────────────────────────────────────────
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

    # ── Inverse: elastic ──────────────────────────────────────────────────────
    # Synthetic ground truth: T300/PESU Ultrason/CAMRI
    #   E_m=3100 MPa, nu_m=0.37,  a11=0.70 a22=0.15 mf=0.25 ar=30
    #   → E1=20654 E2=6338 E3=6257 G12=3287 G13=3259 G23=2230 nu12=0.40 nu13=0.40 nu23=0.42 (MPa)

    # IE1: baseline — GPa measurements, DB material names, "go ahead" included
    "IE1": (
        "I've run tensile and shear tests on my Carbon Fiber T300 / PESU Ultrason specimens "
        "printed on the CAMRI printer. My results: E1=15.45 GPa error of around 0.25 Gpa, E2=5.14 GPa error of around 0.1 GPa, E3=4.21 GPa error of around 0.1 Gpa, "
        "nThe fiber mass fraction is 0.20 and the aspect ratio is 21. Fix the polymer poisson ratio to 0.34"
        "Please run the elastic inverse and report the results "
    ),


    # IE2: measurements in MPa instead of GPa — agent should recognise they're already in MPa
    "IE2": (
        "Elastic characterisation of Carbon Fiber T300 / PESU Ultrason / CAMRI — "
        "my DIC data in MPa: E1=20654 MPa, E2=6338 MPa, E3=6257 MPa, G12=3287 MPa, "
        "nu12=0.40, nu13=0.40.Fiber mass fraction 0.25, aspect ratio 30. "
        "Run the elastic inverse and reportthe results."
    ),

    # IE3: only E1+E2 — underdetermined, identifiability warning expected
    #"IE3": (
    #    "I only have axial tensile data for my T300 / PESU Ultrason / CAMRI "
    #    "specimen: E1=20.65 GPa. Mass fraction is 25%, AR is 30. "
    #    "Please run the elastic inverse and tell me the results"
    #),

# I only have axial tensile data for my T300 / PESU Ultrason / CAMRI specimen: E1=20.65 GPa. Mass fraction is 25%, AR around 30. Please run the elastic inverse and tell me the results
    
# IE3: CT orientation fixed, agent should pass a11/a22 as fixed inputs
    "IE3": (
        "We used micro-CT to characterize the fiber orientation in our T300 / PESU Ultrason / "
        "CAMRI part and measured a11=0.70, a22=0.15. My elastic test results are: "
        "E1=20.65 GPa, E2=6.34 GPa, G12=3.29 GPa, nu12=0.40. Mass fraction 0.25, AR 30. "
        "Use the CT orientation as fixed and run the elastic inverse and report the results"
    ),

    # IE4: ambiguous range — RULE 1 should block tool call, no tool expected
    "IE4": (
        "I have some T300 / PESU Ultrason / CAMRI test data but my measurements have pretty "
        "wide scatter: E1 is roughly 20-21 GPa and E2 is somewhere around 6 GPa. "
        "Can you run the elastic inverse for me?"
    ),

    # ── Inverse: thermoelastic ────────────────────────────────────────────────
    # Self-contained (Path B — no card needed). Ground truth Stage 1 outputs:
    #   T300/PESU Ultrason/CAMRI, matrix_modulus=3100 MPa, matrix_poisson=0.37
    #   a11=0.70, a22=0.15, fiber_massfrac=0.25, ar=30
    # Synthetic CTE targets (f_cte1=-0.7 ppm/K, f_cte2=10 ppm/K, m_cte=55 ppm/K):
    #   CTE11=5.26 ppm/K  CTE22=45.05 ppm/K  CTE33=44.50 ppm/K

    # IT1: CTE in ppm/K — baseline thermoelastic inverse, explicit Stage 1 inputs
    "IT1": (
        "I have CTE DIC results for my T300 / PESU Ultrason / CAMRI specimens: "
        "CTE11=5.26 ppm/K and CTE22=45.05 ppm/K. "
        "From my earlier elastic inverse I know: a11=0.70, a22=0.15, fiber mass fraction=0.25, "
        "aspect ratio=30, matrix modulus=3100 MPa, matrix Poisson ratio=0.37. "
        "Please run the thermoelastic inverse and report the results"
    ),

    # IT2: CTE in 1/K scientific notation + CTE33 included
    "IT2": (
        "Thermoelastic characterisation of T300 / PESU Ultrason / CAMRI. "
        "My  data in 1/K: CTE11=5.26e-6, CTE22=4.505e-5, CTE33=4.450e-5. "
        "Stage 1 results to use as fixed inputs: a11=0.70, a22=0.15, "
        "fiber_massfrac=0.25, AR=30, matrix_modulus=3100 MPa, matrix_poisson=0.37. "
        "Run the thermoelastic inverse — go ahead."
    ),

    # ── Inverse: thermal conductivity ────────────────────────────────────────
    # Self-contained (Path B). Ground truth:
    #   T300/PESU Ultrason/CAMRI,  k_f1=8 W/mK (l2=8), k_f2=1 W/mK (t=8), p1=0.02, p2=0.18
    #   Microstructure from Stage 1: a11=0.70, a22=0.15, mf=0.25, ar=30
    #   Note: solver recovers k_f2 well; l2/t individually may vary (ridge in objective at mf=0.25).
    #   IK2 scored on fit_error convergence (< 0.05), not exact parameter recovery.

    # IK1: wide T range, all 3 channels — should converge cleanly
    "IK1": (
        "I've measured thermal conductivity vs temperature for my T300 / PESU Ultrason "
        "composite printed on the CAMRI printer using laser flash analysis. "
        "All three channels (K11, K22, K33) are available over 25–200°C. "
        "The data file is at "
        "/Users/akshayjacobthomas/Documents/GitHub/DiffMicromehanics/final/eval/data/synthetic_K_wide.csv. "
        "From my Stage 1 elastic inverse: a11=0.70, a22=0.15, fiber mass fraction=0.25, aspect ratio=30. "
        "Fiber is T300, polymer is PESU Ultrason, printer is CAMRI. "
        "Run the thermal inverse and go ahead."
    ),

    # IK2: narrow T range (25–50°C), K11 only — missing channels warning expected
    "IK2": (
        "Thermal conductivity measurement on my Carbon Fiber T300 / PESU Ultrason / CAMRI specimens — "
        "only K11 was measured and the equipment only went up to 50°C. "
        "Data file: "
        "/Users/akshayjacobthomas/Documents/GitHub/DiffMicromehanics/final/eval/data/synthetic_K_narrow.csv. "
        "Stage 1 elastic inverse outputs: a11=0.70, a22=0.15, fiber mass fraction=0.25, aspect ratio=30. "
        "Run the thermal inverse and tell me the results."
    ),

    # ── Forward ───────────────────────────────────────────────────────────────
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
#
# For IE/IT prompts: no gui_value needed. Scored on fit_error from solver output.
# "note" field describes what property the solver should recover (ground truth).

GUI_INPUTS: dict[str, dict] = {
    # ── Inverse: elastic ──────────────────────────────────────────────────────
    "IE1": {
        "goal":  PROMPT_GOALS["IE1"],
        "gui":   "python gui_inverse.py  (Stage 1 — Elastic)",
        "fiber": "Carbon Fiber T300  |  polymer: PESU Ultrason  |  printer: CAMRI",
        "meas":  "E1=15.45 GPa  E2=5.14 GPa  E3=4.21 GPa ",
        "micro": "mf=0.20  AR=21  |  fix nu_m=0.34",
        "fill":  "Read off: a11, a22, matrix_modulus → fill gui_value column.",
    },
    "IE2": {
        "goal":  PROMPT_GOALS["IE2"],
        "gui":   "python gui_inverse.py  (Stage 1 — Elastic)",
        "fiber": "Carbon Fiber T300  |  polymer: PESU Ultrason  |  printer: CAMRI",
        "meas":  "E1=20654 MPa  E2=6338 MPa  E3=6257 MPa  G12=3287 MPa  nu12=0.40  nu13=0.40",
        "micro": "mf=0.25  AR=30  (both as starting guess; leave free)",
        "fill":  "Read off: a11, a22, matrix_modulus → fill gui_value column.",
    },
    "IE3": {
        "goal":  PROMPT_GOALS["IE3"],
        "gui":   "python gui_inverse.py  (Stage 1 — Elastic)",
        "fiber": "Carbon Fiber T300  |  polymer: PESU Ultrason  |  printer: CAMRI",
        "meas":  "E1=20.65 GPa  E2=6.34 GPa  G12=3.29 GPa  nu12=0.40",
        "micro": "mf=0.25  AR=30  |  fix a11=0.70  a22=0.15  (CT-measured orientation)",
        "fill":  "Read off: fiber_massfrac, ar, matrix_modulus, matrix_poisson → fill gui_value column.",
    },
    "IE4": {
        "goal":  PROMPT_GOALS["IE4"],
        "note":  "RULE 1 prompt — agent should NOT call any tool. Pass = tool_correct where expected_tool=None.",
    },
    # ── Inverse: thermoelastic ────────────────────────────────────────────────
    "IT1": {
        "goal":  PROMPT_GOALS["IT1"],
        "gui":   "python gui_inverse.py  (Stage 2 — Thermoelastic)",
        "fiber": "Carbon Fiber T300  |  polymer: PESU Ultrason  |  printer: CAMRI",
        "meas":  "CTE11=5.26e-6 /K  CTE22=45.05e-6 /K",
        "micro": "a11=0.70  a22=0.15  mf=0.25  AR=30  matrix_modulus=3100 MPa  matrix_poisson=0.37",
        "note":  "CTE given in ppm/K in the prompt — agent must convert to 1/K (×1e-6) before passing.",
        "fill":  "Read off: f_cte1_ppm, f_cte2_ppm, m_cte_ppm → fill gui_value column.",
    },
    "IT2": {
        "goal":  PROMPT_GOALS["IT2"],
        "gui":   "python gui_inverse.py  (Stage 2 — Thermoelastic)",
        "fiber": "Carbon Fiber T300  |  polymer: PESU Ultrason  |  printer: CAMRI",
        "meas":  "CTE11=5.26e-6 /K  CTE22=4.505e-5 /K  CTE33=4.450e-5 /K",
        "micro": "a11=0.70  a22=0.15  mf=0.25  AR=30  matrix_modulus=3100 MPa  matrix_poisson=0.37",
        "note":  "CTE already in 1/K — no conversion needed.",
        "fill":  "Read off: f_cte1_ppm, f_cte2_ppm, m_cte_ppm → fill gui_value column.",
    },
    # ── Inverse: thermal ─────────────────────────────────────────────────────
    "IK1": {
        "goal":  PROMPT_GOALS["IK1"],
        "gui":   "python gui_thermal_inverse.py  (Stage 3 — Thermal)",
        "fiber": "Carbon Fiber T300  |  polymer: PESU Ultrason  |  printer: CAMRI",
        "csv":   "eval/data/synthetic_K_wide.csv  (8 points 25–200°C, K11+K22+K33)",
        "micro": "a11=0.70  a22=0.15  mf=0.25  AR=30",
        "fill":  "Read off: k_f1, k_f2, p1, p2 → fill gui_value column.",
    },
    "IK2": {
        "goal":  PROMPT_GOALS["IK2"],
        "note":  "K11 only, 4 points. Tool emits [WARNING — INCOMPLETE CHANNEL DATA]. Pass = agent reports warning.",
    },

    # ── Forward ───────────────────────────────────────────────────────────────
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
# Forward (E/T/K): composite output fields — compared to gui_value via MAPE.
# Inverse (IE): inferred param values only — scored on param MAPE.  No fit_error gate.
# Inverse (IT/IK): fit_error (scored gate) + inferred param values.
#   fit_error gated against _INV_FIT_THRESHOLD; params against _INV_PARAM_THRESHOLD.
# IE4 / IK2: empty list — routing-only.

SCORED_PROPERTIES: dict[str, list[str]] = {
    # Inverse elastic — no fit_error gate; scored on recovered param MAPE only
    "IE1": ["a11", "a22", "matrix_modulus"],
    "IE2": ["a11", "a22", "matrix_modulus"],
    "IE3": ["fiber_massfrac", "ar", "matrix_modulus", "matrix_poisson"],  # a11/a22 fixed from CT
    "IE4": [],              # RULE 1 — no tool expected, routing-only check
    # Inverse thermoelastic — fit_error gate retained
    "IT1": ["fit_error", "f_cte1_ppm", "f_cte2_ppm", "m_cte_ppm"],
    "IT2": ["fit_error", "f_cte1_ppm", "f_cte2_ppm", "m_cte_ppm"],
    # Inverse thermal
    "IK1": ["fit_error", "k_f1", "k_f2", "p1", "p2"],
    "IK2": [],              # missing channel warning check — no fit_error threshold
    # Forward
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
# IE5: None — RULE 1 should block tool call entirely.

from typing import Optional as _Opt
EXPECTED_TOOL: dict[str, _Opt[str]] = {
    # Inverse elastic
    "IE1": "run_elastic_inverse",
    "IE2": "run_elastic_inverse",
    "IE3": "run_elastic_inverse",
    "IE4": None,                       # RULE 1 — no tool expected
    # Inverse thermoelastic
    "IT1": "run_thermoelastic_inverse",
    "IT2": "run_thermoelastic_inverse",
    # Inverse thermal
    "IK1": "run_thermal_inverse",
    "IK2": "run_thermal_inverse",
    # Forward
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
