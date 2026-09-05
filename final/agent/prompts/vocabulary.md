# Vocabulary — Field Name and Unit Mapping

Use this file to map user language to canonical field names and units.
All tools use the canonical names and model units listed here.

---

## Equipment and Systems

| Term | Definition |
|---|---|
| CAMRI | Composites Additive Manufacturing Research Instrument — a medium-scale extrusion deposition additive manufacturing system developed at Purdue University |
| LSAM / Thermwood LSAM | Large-Scale Additive Manufacturing system by Thermwood — a large-format extrusion system with slower cooling rates than CAMRI, leading to different in-situ matrix properties |

### Printer-specific microstructure and constituent expectations

| Printer | a11 typical | a22 typical | a33 typical | In-situ Em | In-situ nu_m | Notes |
|---|---|---|---|---|---|---|
| CAMRI | 0.75–0.85 | 0.10–0.20 | 0.03–0.08 | ~2200–2400 MPa | ~0.33–0.37 | High alignment, near-planar, fast cooling |
| LSAM (Thermwood) | 0.55–0.70 | 0.15–0.30 | 0.08–0.18 | ~1800–1850 MPa | ~0.38–0.42 | Lower alignment due to slower deposition, in-situ matrix softens and nu_m increases vs neat resin |

**Why in-situ Em and nu_m differ between printers:**
The matrix modulus and Poisson ratio of the printed part differ from the neat resin datasheet because the thermal history (cooling rate, residual stress, crystallinity for semi-crystalline polymers) changes during printing. LSAM's slower cooling produces a more compliant, higher-Poisson matrix than CAMRI.
When a Poisson measurement (nu12, nu13) or shear modulus (G12) is available in a transfer, both Em and nu_m are re-inferred together — they are co-identified and cannot be separated individually from a single Poisson measurement.

**Em/nu_m solution manifold (for reporting context):**
Even with E1 + E3 + nu13, the solution is not unique: there is a 1D family of (a11, a22, Em, nu_m) combinations that all fit the measurements exactly. The optimizer converges to one point on this family; the Em result (typically 1800–1835 MPa for LSAM/PESU) is the attractor of the gradient descent, not a uniquely determined value. Adding G12 would fully constrain the system.

---

## Microstructure

| User says | Canonical field | Units | Notes |
|---|---|---|---|
| fiber alignment, orientation, fiber orientation, anisotropy | a11, a22, a12, a13, a23 | dimensionless | a33 = 1 − a11 − a22, inferred automatically |
| alignment along print direction, axial alignment | a11 | dimensionless | typical range 0.3–0.9 |
| transverse alignment, in-plane transverse | a22 | dimensionless | typical range 0.05–0.4 |
| fiber mass fraction, fiber loading, fiber content, wf, weight fraction | fiber_massfrac | dimensionless | typical range 0.01–0.60 |
| aspect ratio, fiber length, l/d ratio, slenderness | ar | dimensionless | typical range 5–100 |

### Orientation shorthand — resolve BEFORE calling any tool

| User says | a11 | a22 | a33 | Notes |
|---|---|---|---|---|
| random, random orientation, randomly oriented, 3D random, 3D isotropic | 0.333 | 0.333 | 0.333 | 3D random — fibers equally distributed in all directions |
| 2D random, in-plane random, planar random, planar isotropic, in-plane isotropic, isotropic | 0.5 | 0.5 | 0.0 | Fibers randomly distributed within a plane |
| aligned, unidirectional, UD, fully aligned | 1.0 | 0.0 | 0.0 | All fibers along print direction |
| transverse, cross-ply | 0.0 | 1.0 | 0.0 | All fibers perpendicular to print direction |

**Rule:** If the user gives orientation as a keyword (e.g. "random"), look up the exact values above and pass them directly. Never invent or approximate orientation values. If the user gives a keyword alongside an explicit a33 value that contradicts the keyword's default (e.g. "planar isotropic with a33=0.1"), flag the contradiction explicitly — e.g. "Note: 'planar isotropic' implies a33=0.0, which conflicts with your stated a33=0.1. I will use the planar isotropic defaults (a11=a22=0.5, a33=0.0)." Then proceed with the keyword's table values.

---

## Elastic composite properties

| User says | Canonical field | Model unit | Conversion |
|---|---|---|---|
| axial modulus, modulus in print direction, longitudinal stiffness, E along x | E1 | MPa | if user gives GPa: × 1000 |
| transverse modulus, in-plane transverse stiffness | E2 | MPa | if user gives GPa: × 1000 |
| out-of-plane modulus | E3 | MPa | if user gives GPa: × 1000 |
| in-plane shear modulus, shear stiffness | G12 | MPa | if user gives GPa: × 1000 |
| in-plane Poisson ratio, Poisson, lateral contraction | nu12 | dimensionless | — |

---

## Thermoelastic composite properties

| User says | Canonical field | Model unit | Conversion |
|---|---|---|---|
| thermal expansion in print direction, alpha 1, axial CTE, longitudinal CTE | CTE11 | 1/K | if user gives ppm/K: × 1e-6 |
| transverse CTE, alpha 2, in-plane transverse thermal expansion | CTE22 | 1/K | if user gives ppm/K: × 1e-6 |
| through-thickness CTE, alpha 3 | CTE33 | 1/K | if user gives ppm/K: × 1e-6 |

---

## Thermal composite properties

| User says | Canonical field | Model unit |
|---|---|---|
| thermal conductivity along print direction, axial conductivity, k parallel | K11 | W/m·K |
| transverse conductivity, k transverse | K22 | W/m·K |
| through-thickness conductivity | K33 | W/m·K |

---

## Constituent properties

| User says | Canonical field | Model unit | Notes |
|---|---|---|---|
| matrix stiffness, polymer modulus, resin modulus, matrix Young's modulus | matrix_E | MPa | inferred in Stage 1 — in-situ value often differs from datasheet |
| matrix Poisson ratio, polymer Poisson | matrix_nu | dimensionless | inferred in Stage 1 |
| fiber axial CTE, fiber longitudinal thermal expansion, alpha fiber 1 | f_cte1 | 1/K | inferred in Stage 2 — datasheet values unreliable, always infer |
| fiber transverse CTE, alpha fiber 2 | f_cte2 | 1/K | inferred in Stage 2 |
| matrix CTE, polymer CTE, resin CTE | m_cte | 1/K | inferred in Stage 2 |
| fiber axial conductivity, fiber longitudinal conductivity, k fiber axial, k_f1 | k_f1_WmK | W/m·K | forward prediction input — pass to predict_thermal_conductivity |
| fiber transverse conductivity, k fiber transverse, k_f2 | k_f2_WmK | W/m·K | forward prediction input — pass to predict_thermal_conductivity |
| matrix conductivity, polymer conductivity, resin conductivity, k_m, k matrix | k_m_WmK | W/m·K | scalar matrix conductivity for forward prediction |
| fiber longitudinal conductivity (inferred) | k_l2 | W/m·K | inferred in Stage 3 |
| polymer conductivity slope, k matrix p1 | k_p1 | W/m·K | parametric model coefficient — K_m(T) = k_p1*sqrt(T/T_ref) + k_p2 |
| polymer conductivity intercept, k matrix p2 | k_p2 | W/m·K | parametric model coefficient |

---

## Typical value ranges (for flagging implausible inputs)

| Field | Typical range | Flag if outside |
|---|---|---|
| E1 | 5,000–150,000 MPa | < 1,000 or > 300,000 |
| E2 | 3,000–30,000 MPa | < 500 or > 80,000 |
| G12 | 1,000–15,000 MPa | < 200 or > 40,000 |
| nu12 | 0.05–0.45 | < 0 or > 0.5 |
| CTE11 | −5e-6 to 20e-6 1/K | outside −20e-6 to 50e-6 |
| CTE22 | 10e-6 to 80e-6 1/K | outside 1e-6 to 200e-6 |
| K11 | 0.5–20 W/m·K | < 0.1 or > 100 |
| fiber_massfrac | 0.10–0.50 | < 0.01 or > 0.70 |
| ar | 5–50 | < 1 or > 500 |
| a11 | 0.30–0.90 | < 0 or > 1 |
