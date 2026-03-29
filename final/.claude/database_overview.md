# DiffMicromechanics — Database Overview

## The core idea in one sentence

A composite is made of a **fiber + polymer + printer**. The DB separates what
belongs to the *material* (intrinsic, printer-agnostic) from what belongs to the
*printed part* (printer-specific).

---

## Three-layer separation

```
┌─────────────────────────────────────────────────────────────────┐
│  CONSTITUENT LAYER  (intrinsic — same regardless of printer)    │
│                                                                 │
│   fibers              polymers                                  │
│   ─────────────       ─────────────                             │
│   E1, E2, G12         E1, nu12                                  │
│   CTE1, CTE2          CTE                                       │
│   k_f1, k_f2          k_m, p1, p2                               │
│   (datasheet values)  (datasheet values)                        │
└───────────────────────┬─────────────────────────────────────────┘
                        │ combined with
┌───────────────────────▼─────────────────────────────────────────┐
│  MICROSTRUCTURE LAYER  (printer-specific — changes per machine) │
│                                                                 │
│   orientation tensor:  a11, a22, a12, a13, a23                  │
│   morphology:          mass fraction (mf), aspect ratio (ar)    │
└───────────────────────┬─────────────────────────────────────────┘
                        │ produces
┌───────────────────────▼─────────────────────────────────────────┐
│  COMPOSITE LAYER  (derived — predicted or measured)             │
│                                                                 │
│   elastic:       E1, E2, E3, G12, nu12, …                       │
│   thermoelastic: CTE11, CTE22, …                                │
│   thermal:       k11, k22, k33  (vs temperature)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## All 11 tables at a glance

### Reference tables — seeded once, never overwritten by solvers

| Table | What it stores | Written by |
|---|---|---|
| `fibers` | Datasheet properties of fiber types (T300, AS4, E-Glass) | `init_db.py` + JSON |
| `polymers` | Datasheet properties of matrix polymers (PESU, Epoxy) | `init_db.py` + JSON |
| `printers` | Printer registry (Markforged X7, Anisoprint, …) | User via GUI or Python |

### Material card tables — written during the solver workflow

| Table | What it stores | One row = |
|---|---|---|
| `print_configs` | A (fiber, polymer, printer) triple with a name | One material card |
| `microstructure_snapshots` | Orientation tensor + morphology at a point in time | One snapshot per solve |
| `inference_runs` | Full audit trail of every solver run | One optimisation run |
| `constituent_property_values` | Inferred or inputted material-level properties | One property value |
| `experimental_measurements` | User-measured composite properties with uncertainty | One measurement |
| `composite_property_values` | Surrogate-predicted composite properties | One predicted value |
| `property_preferences` | User's preferred source per property per card | One preference override |
| `current_composite_properties` | Cached current best value per property per card | One resolved value |

---

## What a "material card" is

A **material card** = one `print_configs` row + everything linked to it.

```
print_configs  (the card)
│
├── microstructure_snapshots    how the fiber was oriented in this printer
│
├── inference_runs              every time the solver ran on this card
│
├── constituent_property_values inferred material properties (matrix E, fiber CTE, k)
│
├── experimental_measurements   what the user measured on printed coupons
│
├── composite_property_values   what the surrogate predicted
│
└── current_composite_properties  ← resolved "current best" value per property
```

---

## Provenance — every value knows where it came from

Every stored value carries a `source_tag`:

| Tag | Meaning |
|---|---|
| `web` | From a datasheet or literature (seeded in JSON) |
| `inputted` | User typed it in manually |
| `inferred` | Recovered by the inverse solver |
| `predicted` | Output of the forward surrogate |

When loading a card, the resolution hierarchy is:

```
experimental  >  inferred  >  predicted  >  inputted  >  web (neat)
```

Higher priority wins. The full history is always kept — nothing is overwritten.

---

## Global vs card-scoped constituent properties

`constituent_property_values` has a nullable `print_config_id`:

| `print_config_id` | Meaning | Example |
|---|---|---|
| `NULL` (global) | Intrinsic to the material — reusable across all cards with this fiber/polymer | Matrix E, fiber CTE, k_f1, k_f2, k_m, p1, p2 |
| Non-NULL (card-scoped) | Specific to one (fiber, polymer, printer) combination | (reserved for rare processing-dependent overrides) |

**Key insight:** infer matrix E from Printer A data → save globally → load into
Printer B card automatically. No re-inference needed for the same resin.

---

## Information flow through the four-stage workflow

### Stage 1 — Elastic inverse

**Input:** measured E1, E2, G12, nu12 from printed coupon
**Free variables:** matrix_modulus, matrix_poisson, a11, a22, a12, a13, a23, mf, ar
**Fixed:** fiber datasheet properties (from `fibers` table)

**Writes to DB:**
```
inference_runs          ← stage='elastic', full input/output JSON, loss
microstructure_snapshots ← a11, a22, ..., mf, ar  (provenance: inferred)
constituent_property_values ← matrix_modulus, matrix_poisson
                              (constituent_type='polymer', print_config_id=NULL)
composite_property_values   ← predicted E1, E2, G12, nu12
experimental_measurements   ← measured E1, E2, G12, nu12 (the targets)
```

---

### Stage 2 — Thermoelastic inverse

**Input:** measured CTE11, CTE22
**Free variables:** f_CTE1, f_CTE2, matrix_CTE
**Fixed:** microstructure + matrix_modulus/poisson loaded from Stage 1 card

**Writes to DB:**
```
inference_runs          ← stage='thermoelastic'
constituent_property_values ← f_CTE1, f_CTE2 (fiber, NULL)
                              matrix_CTE      (polymer, NULL)
experimental_measurements   ← measured CTE11, CTE22
```

---

### Stage 3 — Thermal inverse

**Input:** measured k11, k22, k33 vs temperature (CSV)
**Free variables:** l2 (→k_f1), t (→k_f2), p1, p2
**Fixed:** microstructure from Stage 1 card

**Writes to DB:**
```
inference_runs          ← stage='thermal_inverse'
constituent_property_values ← k_f1 = l2         (fiber,   NULL)
                              k_f2 = l2/t        (fiber,   NULL)
                              p1, p2             (polymer, NULL)
```

---

### Stage 4 — Forward prediction on a new printer

**Input:** new printer's orientation tensor (measured or estimated)
**Auto-filled from DB:** all inferred constituent properties from Stages 1–3

**Writes to DB:**
```
print_configs           ← new card for Printer B
microstructure_snapshots ← Printer B orientation (provenance: inputted)
inference_runs          ← stage='elastic' (or thermoelastic/thermal)
composite_property_values ← predicted E1, CTE11, k11, … for Printer B
```

---

## How the same inferred properties flow across cards

```
Printer A card                      Printer B card
─────────────                       ─────────────
 Stage 1 solve                       Stage 4 forward run
   │                                   │
   ▼                                   ▼
constituent_property_values  ──────►  loaded via get_constituent_properties()
  matrix_modulus = 3450 MPa            matrix_modulus = 3450 MPa  (same row)
  f_CTE1 = -0.5e-6 /K                 f_CTE1 = -0.5e-6 /K
  k_f1 = 10.2 W/m·K                   k_f1 = 10.2 W/m·K

microstructure_snapshots              microstructure_snapshots
  a11=0.77 (Printer A)                  a11=0.55 (Printer B)  ← different

composite_property_values             composite_property_values
  E1=52 GPa (Printer A result)          E1=38 GPa (Printer B prediction)
```

The constituent properties are shared (one row, referenced by both cards).
The microstructure and composite outputs are card-specific.

---

## What happens if you infer the same property twice?

Every save is an **INSERT** — nothing is ever overwritten. Both rows are kept.

```
constituent_property_values
─────────────────────────────────────────────────────────────────────
id  constituent  property       value    source_tag  inference_run_id  created_at
─────────────────────────────────────────────────────────────────────
 5  polymer      matrix_modulus  3450    inferred     run 3 (Printer A)  2026-02-10
12  polymer      matrix_modulus  3380    inferred     run 9 (Printer B)  2026-03-15
```

On load: **most recent wins** (3380). The full history is always queryable.
Future improvement: a `prefer_run_id` parameter on load to pin a specific value.

---

## ER diagram (simplified)

```
fibers ──────────────────────────────────────────────┐
                                                      │
polymers ────────────────────────────────────────┐   │
                                                  │   │
printers ─────────────────────┐                  │   │
                               │                  │   │
                               ▼                  ▼   ▼
                          print_configs  ◄─────────────────
                               │
          ┌────────────────────┼──────────────────────┐
          ▼                    ▼                       ▼
microstructure_         inference_runs         experimental_
snapshots                      │               measurements
                               │
              ┌────────────────┼─────────────────────┐
              ▼                ▼                      ▼
  constituent_property_  composite_property_   current_composite_
  values                 values                properties
                                                (cache / resolved winner)
                                    ▲
                         property_preferences
                         (source override per card)
```
