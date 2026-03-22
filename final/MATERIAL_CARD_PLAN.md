# Material Card System — Design Plan
## CF-PEI Composite Database with Provenance Tracking

---

## 1. Core Design Insight

The workflow has a fundamental three-layer separation that the current schema does not express:

| Layer | What it is | Printer-dependent? |
|---|---|---|
| **Constituent** | Fiber E, CTE, k / Matrix E, CTE, k | No — CF is CF regardless of printer |
| **Microstructure** | Orientation tensor, mf, aspect ratio | Yes — the printer determines fiber alignment |
| **Composite** | E1, CTE11, k11, ... | Yes — derived from the two layers above |

The current schema conflates all three into flat columns in `microstructure` and `inferred_properties`. Everything needs to be restructured around this separation.

---

## 2. The `print_config` — Material Card Identity

A **material card** is uniquely identified by **(fiber, polymer, printer)**. Every inference run, experimental measurement, and predicted property attaches to a `print_config` row. The card *evolves* as you add more data — it is never overwritten, it accumulates.

```
print_configs
  id  |  fiber_id  |  polymer_id  |  printer_id    |  name
   1       CF-PEI       PEI          Markforged X7    "CF-PEI / MFX7"
   2       CF-PEI       PEI          Anisoprint        "CF-PEI / AP"
```

Constituent properties (fiber CTE, matrix modulus) belong to the fiber/polymer rows — **not** to the print_config. The print_config only owns microstructure and composite-level data.

---

## 3. Provenance Tags

Every stored value carries one of four tags:

| Tag | Meaning |
|---|---|
| `web` | From a datasheet or literature source |
| `inputted` | User typed it manually |
| `inferred` | Output of an inverse solver run |
| `predicted` | Output of a forward solver run |

---

## 4. Complete Database — All Tables Explained

The database is a single SQLite file (`data/micromechanics.db`). All tables below live in that one file, linked by foreign keys.

---

### 4.1 `fibers` ← existing, extended

**What it is**: One row per fiber type in the material library. Stores the neat (pure, unreinforced) fiber properties from datasheets. These values never change from inference — they are web/inputted baselines only.

**New columns added** (CTE and thermal conductivity, previously missing):

```sql
neat_CTE1        REAL   -- µ/K  axial CTE (from datasheet)
neat_CTE2        REAL   -- µ/K  transverse CTE
neat_k1          REAL   -- W/m·K  longitudinal thermal conductivity
neat_k2          REAL   -- W/m·K  transverse thermal conductivity
neat_CTE_source  TEXT   -- e.g. "Toray datasheet", "Soden et al. 1998"
neat_k_source    TEXT
```

**Seeded from**: `data/fibers.json` via `init_db.py`.
**Never written to by inference** — inferred constituent values go to `constituent_property_values` instead.

---

### 4.2 `polymers` ← existing, extended

**What it is**: Same as `fibers` but for the matrix material. One row per polymer.

**New columns added**:

```sql
neat_CTE         REAL   -- µ/K
neat_k           REAL   -- W/m·K
neat_CTE_source  TEXT
neat_k_source    TEXT
```

**Seeded from**: `data/polymers.json`.
**Never written to by inference**.

---

### 4.3 `printers` ← existing, unchanged

**What it is**: One row per printer machine. Currently empty — no seed data exists yet.

```sql
id           INTEGER PRIMARY KEY
name         TEXT     -- e.g. "Markforged X7"
manufacturer TEXT
notes        TEXT
```

**Connected to**: `print_configs` (one printer can have many cards).

---

### 4.4 `print_configs` ← NEW

**What it is**: The material card anchor. One row = one specific material system (fiber + polymer) on one specific printer. Everything else in the schema hangs off this table.

```sql
CREATE TABLE print_configs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,       -- e.g. "CF-PEI / Markforged X7"
    fiber_id    INTEGER NOT NULL REFERENCES fibers(id),
    polymer_id  INTEGER NOT NULL REFERENCES polymers(id),
    printer_id  INTEGER          REFERENCES printers(id),
    notes       TEXT,
    created_at  TEXT               -- ISO-8601
);
```

**Connected to**: every other new table points back here via `print_config_id`.

**Example rows**:
```
id | fiber_id | polymer_id | printer_id | name
 1      1           1           1         "CF-PEI / Markforged X7"
 2      1           1           2         "CF-PEI / Anisoprint"
```

Same fiber and polymer, two different printers = two separate cards, each with their own microstructure and composite property history.

---

### 4.5 `microstructure_snapshots` ← NEW (replaces `microstructure`)

**What it is**: The printer-specific morphology, versioned. Every time you infer or manually enter an orientation tensor, a **new row is added** — nothing is ever overwritten. The most recent row is the current microstructure for that card. Mass fraction (`mf`) is the single fraction field — this matches the `w_f` field used throughout surrogate training. The mapping from `mf` (database) → `w_f` (model input) is handled in `db.py`.

```sql
CREATE TABLE microstructure_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    print_config_id  INTEGER NOT NULL REFERENCES print_configs(id),
    mf               REAL,   -- mass fraction (w_f in model inputs)
    ar               REAL,
    a11 REAL, a22 REAL, a12 REAL, a13 REAL, a23 REAL,
    provenance_json  TEXT,    -- per-field source tag (see note below)
    inference_run_id INTEGER  REFERENCES inference_runs(id),
    notes            TEXT,
    created_at       TEXT
);
```

**Why `provenance_json` instead of a single `source_tag`**:

A single source tag per row assumes every field came from the same place — which is not true in practice. For example:

- `mf = 0.20` — inputted (you weighed the sample)
- `ar = 20.0` — web (from fiber datasheet)
- `a11 = 0.77` — inferred (from elastic inverse solve)
- `a22 = 0.16` — inferred (from elastic inverse solve)

`provenance_json` is a fixed-size JSON column on each row that records the source of each field individually:

```json
{
  "mf":  {"source_tag": "inputted"},
  "ar":  {"source_tag": "web"},
  "a11": {"source_tag": "inferred", "inference_run_id": 3},
  "a22": {"source_tag": "inferred", "inference_run_id": 3},
  "a12": {"source_tag": "inputted"},
  "a13": {"source_tag": "inputted"},
  "a23": {"source_tag": "inputted"}
}
-- 7 fields total: mf, ar, a11, a22, a12, a13, a23
```

This column does **not** grow over time — it has exactly 8 entries describing the 8 fields in that row, set at creation and never changed. What grows is the number of rows in the table (one per snapshot event).

**Example rows for one card**:
```
id | print_config_id | a11  | a22  | created_at   | provenance_json (simplified)
 1       1             0.60   0.10   2026-01-10     {a11: inputted, a22: inputted}
 2       1             0.77   0.16   2026-02-15     {a11: inferred, a22: inferred}
 3       1             0.74   0.18   2026-03-21     {a11: inferred, a22: inferred}
```

Row 1 = initial manual guess. Row 2 = after first Stage 1 solve. Row 3 = after re-running Stage 1 with better data. All three are preserved. The GUI uses the most recent row (row 3) as the current microstructure.

**Connected to**: `print_configs` (many snapshots per card), `inference_runs` (optional — which run produced this).

---

### 4.6 `inference_runs` ← NEW (replaces `experiments` + `inferred_properties`)

**What it is**: One row per solver invocation. A complete, frozen audit record of a solve — what went in, what came out, what settings were used. Replaces both the old `experiments` and `inferred_properties` tables.

```sql
CREATE TABLE inference_runs (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    print_config_id        INTEGER NOT NULL REFERENCES print_configs(id),
    stage                  TEXT NOT NULL,
    -- 'elastic' | 'thermoelastic' | 'thermal_inverse' | 'thermal_forward'
    microstructure_snap_id INTEGER REFERENCES microstructure_snapshots(id),
    -- which microstructure was used as input to this run
    inputs_json            TEXT,   -- full snapshot of every input value
    outputs_json           TEXT,   -- full snapshot of every output value
    solver_json            TEXT,   -- solver settings (method, maxiter, tol, etc.)
    loss                   REAL,
    notes                  TEXT,
    created_at             TEXT
);
```

`inputs_json` and `outputs_json` mean you can always reconstruct exactly what was run, even years later.

**Connected to**:
- `print_configs` — which card this run belongs to
- `microstructure_snapshots` — which microstructure it used (and also which snapshots it produced)
- `constituent_property_values` — the inferred constituent values produced by this run
- `composite_property_values` — the predicted composite values produced by this run

---

### 4.7 `constituent_property_values` ← NEW

**What it is**: Where inferred constituent properties accumulate. This is where Stage 1 (matrix E, nu) and Stage 2 (fiber CTE, matrix CTE) results are stored. Importantly, these values are **printer-agnostic** — fiber CTE is an intrinsic property of the fiber regardless of which printer was used.

```sql
CREATE TABLE constituent_property_values (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    constituent_type TEXT NOT NULL,    -- 'fiber' | 'polymer'
    constituent_id   INTEGER NOT NULL, -- → fibers.id or polymers.id
    print_config_id  INTEGER REFERENCES print_configs(id),
    -- NULL = global (not printer-specific, reusable across all cards)
    property_name    TEXT NOT NULL,
    -- 'E1','E2','G12','nu12','CTE1','CTE2','k1','k2', 'p1','p2','l2','t' etc.
    value            REAL NOT NULL,
    unit             TEXT,
    source_tag       TEXT NOT NULL,    -- 'web' | 'inputted' | 'inferred' | 'predicted'
    inference_run_id INTEGER REFERENCES inference_runs(id),
    notes            TEXT,
    created_at       TEXT
);
```

**The `print_config_id = NULL` rule**: When Stage 2 infers `fiber_CTE1` from the Markforged card, it is stored with `print_config_id = NULL` — meaning it is globally available. When you later set up a new card for Printer B, `get_canonical_value()` finds this row and pre-fills the field automatically. You do not need to re-run Stage 2 for every printer.

**Multiple inferred values for the same property**: If you run Stage 1 five times, you get five rows for `matrix_modulus`. All are kept. The resolution logic picks the most recent by default (see Section 5 — Value Resolution).

**Example rows**:
```
id | constituent_type | constituent_id | property_name | value  | source_tag | print_config_id
 1      polymer              1              E1            3500    web           null
 2      polymer              1              E1            3200    inferred        1       ← Stage 1, MFX7
 3      fiber                1              CTE1          0.30    inferred       null     ← Stage 2, global
 4      polymer              1              CTE           52.0    inferred       null     ← Stage 2, global
```

**Connected to**: `fibers` / `polymers` (which constituent), `print_configs` (nullable), `inference_runs` (which run produced it).

---

### 4.8 `composite_property_values` ← NEW

**What it is**: All composite-level outputs for a given card — both predicted (from forward/inverse solves) and experimental (from measurements). Predicted and experimental values coexist here, distinguished by `source_tag`. `property_preferences` controls which one the GUI treats as canonical.

```sql
CREATE TABLE composite_property_values (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    print_config_id  INTEGER NOT NULL REFERENCES print_configs(id),
    property_name    TEXT NOT NULL,
    -- 'E1','E2','G12','nu12','CTE11','CTE22','CTE33','k11','k22','k33' etc.
    value            REAL NOT NULL,
    unit             TEXT,
    source_tag       TEXT NOT NULL,  -- 'predicted' | 'experimental' | 'inputted'
    inference_run_id INTEGER REFERENCES inference_runs(id),
    measurement_id   INTEGER REFERENCES experimental_measurements(id),
    temperature_C    REAL,           -- nullable; for thermal/CTE data at a given temp
    created_at       TEXT
);
```

**Connected to**: `print_configs`, `inference_runs` (if predicted), `experimental_measurements` (if experimental).

---

### 4.9 `experimental_measurements` ← NEW

**What it is**: Raw experimental data entered by the user. Completely separate from predictions. These are the ground-truth targets that drive the inverse solver. All property types — mechanical, CTE, thermal conductivity — live in one table, distinguished by `property_name`. Temperature-dependent data (k vs. T, CTE vs. T) uses the `temperature_C` column — one row per temperature per direction.

```sql
CREATE TABLE experimental_measurements (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    print_config_id  INTEGER NOT NULL REFERENCES print_configs(id),
    property_name    TEXT NOT NULL,   -- 'E1', 'CTE11', 'k11', etc.
    value            REAL NOT NULL,
    unit             TEXT,
    uncertainty      REAL,            -- measurement uncertainty / σ, same unit as value
    temperature_C    REAL,            -- test temperature (nullable)
    conditions_json  TEXT,            -- other test conditions as JSON
    reference        TEXT,            -- citation or lab notebook ref
    date             TEXT,
    notes            TEXT
);
```

**Why `uncertainty` is here**: The inverse GUI already has a σ field per target output that feeds the ε-insensitive loss. Previously σ was discarded after the solve. It now lives alongside the measurement it belongs to.

**Example rows for k vs. T data**:
```
id | print_config_id | property_name | value | unit   | temperature_C | uncertainty
 1       1                k11          0.42    W/m·K       25             0.02
 2       1                k11          0.44    W/m·K       50             0.02
 3       1                k11          0.46    W/m·K       75             0.02
 4       1                k22          0.38    W/m·K       25             0.02
```

**Connected to**: `print_configs`, `composite_property_values` (a measurement can be referenced by a composite property row).

---

### 4.10 `property_preferences` ← NEW

**What it is**: Per card, per property — which source the user wants treated as canonical. Only needs a row when overriding the default resolution order. If no row exists for a property, the default kicks in automatically.

```sql
CREATE TABLE property_preferences (
    print_config_id  INTEGER NOT NULL REFERENCES print_configs(id),
    property_name    TEXT NOT NULL,
    preferred_source TEXT NOT NULL,
    -- 'experimental' | 'predicted' | 'inferred' | 'web' | 'inputted'
    PRIMARY KEY (print_config_id, property_name)
);
```

**Example**: You trust your experimental k11 measurement more than the surrogate prediction for the Markforged card. You set `preferred_source = 'experimental'` for `k11` on card 1. That one row is all that is needed — everything else continues to use the default resolution.

**Connected to**: `print_configs`.

---

## 5. How Tables Connect — Full Diagram

```
fibers ──────────────────────────────────────────────────────┐
polymers ────────────────────────────────────────────────────┤
printers ────────────────────────────────────────────────────┤
                                                             ▼
                                                      print_configs
                                                             │
              ┌──────────────────┬──────────────────┬───────┴──────────────┐
              ▼                  ▼                  ▼                       ▼
  microstructure_snapshots  inference_runs  experimental_measurements  property_preferences
              │                  │                  │
              │            ┌─────┴──────┐           │
              │            ▼            ▼           │
              │  constituent_     composite_         │
              │  property_values  property_values ◄──┘
              │            ▲            ▲
              └────────────┘            │
          (snapshot used               (measurement
           as input to run)             referenced)
```

**Reading the diagram**:
- `print_configs` is the hub — everything attaches to it
- `inference_runs` is the central event record — it connects to the snapshot used as input, and is referenced by both `constituent_property_values` and `composite_property_values` to show which run produced which value
- `experimental_measurements` feeds into `composite_property_values` (a measured value can be stored as a composite property with `source_tag='experimental'`)
- `fibers`, `polymers`, `printers` are reference tables — they don't change from inference, only from manual updates to the seed JSON files

---

## 6. Value Resolution — When Multiple Values Exist

When the GUI asks for a canonical value (e.g. "what is matrix_modulus for PEI on card 1?"), the following order applies:

```
1. property_preferences override    ← user has explicitly pinned a source
2. inputted                         ← user typed a value in the current session
3. inferred (most recent)           ← from constituent_property_values
4. web                              ← fibers.neat_* / polymers.neat_* columns
5. predicted                        ← rarely used for constituent properties
```

**Why most recent, not lowest loss**: Raw loss is not comparable across runs that used different target sets (a run targeting 1 output will have lower loss than one targeting 4, even if the 4-output run is more reliable). Most recent is safer as a default. The user can always pin a specific run via `property_preferences`.

**The `print_config_id = NULL` lookup**: `get_canonical_value()` first checks for `constituent_property_values` rows scoped to the specific `print_config_id`, then falls back to rows with `print_config_id = NULL` (global), then falls back to the fiber/polymer table columns.

---

## 7. Inference Pipeline Flow

```
                  ┌─────────────────────────────────────────────────┐
                  │       print_config: CF-PEI / Markforged X7       │
                  └─────────────────────────────────────────────────┘
                                         │
         ┌───────────────────────────────┼──────────────────────────────┐
         ▼                               ▼                              ▼
  constituent_property_values   microstructure_snapshots    composite_property_values
  (printer-agnostic inferred)   (printer-specific)          (predicted + experimental)
         ▲                               ▲                              ▲
         │                               │                              │
  ┌──────┴──────┐                ┌───────┴──────┐               ┌──────┴──────┐
  │  Stage 1    │                │  Stage 1     │               │  Stage 1    │
  │  Elastic    │                │  Elastic     │               │  Elastic    │
  │  Inverse    │                │  Inverse     │               │  prediction │
  │             │                │              │               │             │
  │  infers:    │                │  infers:     │               │  E1,E2,G12  │
  │  matrix_E   │                │  a11,a22,ar  │               │  nu12,...   │
  │  matrix_nu  │                │  mf          │               │             │
  └──────┬──────┘                └──────────────┘               └─────────────┘
         │
         │  inferred matrix_E and matrix_nu are now fixed for Stage 2
         ▼
  ┌──────────────┐
  │   Stage 2    │   uses fixed microstructure snapshot from Stage 1
  │   Thermoel.  │
  │   Inverse    │
  │              │
  │   infers:    │
  │   fiber_CTE1 │  stored with print_config_id=NULL
  │   fiber_CTE2 │  (printer-independent, reusable across all cards)
  │   matrix_CTE │
  └──────┬───────┘
         │
         │  fiber_CTE and matrix_CTE now globally known for CF-PEI
         ▼
  ┌──────────────────────────────────────────────────────┐
  │  New card: CF-PEI / Anisoprint (Printer B)           │
  │                                                      │
  │  Stage 3 — Thermal Forward                           │
  │                                                      │
  │  Loads automatically:                                │
  │    fiber_k1, fiber_k2    ← from fibers table (web)   │
  │    matrix_k              ← from polymers table (web) │
  │    fiber_CTE1/CTE2       ← from Stage 2 (inferred)   │
  │    matrix_CTE            ← from Stage 2 (inferred)   │
  │                                                      │
  │  Only needs:                                         │
  │    a11, a22, mf, ar      ← Printer B specific        │
  │                          ← inputted or measured      │
  │                                                      │
  │  Predicts:                                           │
  │    k11, k22, k33         ← saved as composite props  │
  └──────────────────────────────────────────────────────┘
```

---

## 8. Stage-by-Stage Workflow Detail

### Stage 1 — Elastic Inverse

**Goal**: Infer microstructure (a11, a22, ar, mf) and matrix properties (E, nu) from experimental composite moduli.

| Action | Where stored |
|---|---|
| User enters experimental E1, E2, G12, nu12 + uncertainties | `experimental_measurements` |
| Fiber mechanical props (E, nu) loaded from datasheet | `fibers.neat_*` — read only |
| Solver runs; infers matrix_E, matrix_nu | `constituent_property_values` (source_tag='inferred') |
| Solver runs; infers a11, a22, ar, mf | `microstructure_snapshots` (provenance_json marks these as 'inferred') |
| Forward prediction at inferred inputs | `composite_property_values` (source_tag='predicted') |
| Full solver audit record | `inference_runs` (stage='elastic') |

---

### Stage 2 — Thermoelastic Inverse

**Goal**: Infer fiber CTE and matrix CTE from experimental composite CTEs. Microstructure from Stage 1 is fixed.

| Action | Where stored |
|---|---|
| User enters experimental CTE11, CTE22 + uncertainties | `experimental_measurements` |
| System loads Stage 1 microstructure snapshot | `microstructure_snapshots` — read only |
| System loads Stage 1 inferred matrix_E, matrix_nu | `constituent_property_values` — used as fixed inputs |
| Solver runs; infers fiber_CTE1, fiber_CTE2, matrix_CTE | `constituent_property_values` with print_config_id=NULL |
| Forward CTE prediction | `composite_property_values` (source_tag='predicted') |
| Full solver audit record | `inference_runs` (stage='thermoelastic') |

---

### Stage 3 — Thermal Forward / Inverse

**Goal**: Predict or infer composite thermal conductivities.

**Forward path** (for Printer B — no experimental k needed):

| Action | Where stored |
|---|---|
| System auto-loads fiber_k1, fiber_k2 | `fibers.neat_k*` — read only |
| System auto-loads matrix_k | `polymers.neat_k` — read only |
| User inputs Printer B orientation (a11, a22, mf, ar) | `microstructure_snapshots` (provenance_json: 'inputted') |
| Forward prediction runs | `composite_property_values` (source_tag='predicted') |
| Audit record | `inference_runs` (stage='thermal_forward') |

**Inverse path** (if experimental k values exist):

| Action | Where stored |
|---|---|
| User loads k11, k22, k33 vs. temperature CSV | `experimental_measurements` (one row per temperature per direction) |
| Solver infers p1, p2, l2, t (constituent k parameters) | `constituent_property_values` (source_tag='inferred') |
| Audit record | `inference_runs` (stage='thermal_inverse') |

---

## 9. `db.py` API — New Functions Required

```python
# --- print_configs (material cards) ---
create_print_config(fiber_id, polymer_id, printer_id, name, notes="") -> int
get_print_config(id) -> dict
get_all_print_configs() -> list[dict]

# --- microstructure snapshots ---
save_microstructure_snapshot(print_config_id, mf, ar, a11, a22, a12, a13, a23,
                              provenance, inference_run_id=None, notes="") -> int
# provenance: dict mapping field names to {source_tag, inference_run_id}
# e.g. {"a11": {"source_tag": "inferred", "inference_run_id": 3}, "mf": {"source_tag": "inputted"}}
# note: mf maps to w_f in surrogate model inputs — conversion handled in db.py
get_latest_microstructure(print_config_id) -> dict   # most recent row
get_best_microstructure(print_config_id) -> dict     # lowest-loss inference run

# --- experimental measurements ---
save_experimental_measurement(print_config_id, property_name, value, unit,
                               uncertainty=None, temperature_C=None,
                               conditions=None, reference="",
                               date=None, notes="") -> int
get_experimental_measurements(print_config_id,
                               property_name=None) -> list[dict]

# --- inference runs ---
save_inference_run(print_config_id, stage, inputs_json, outputs_json,
                   loss, solver_json, microstructure_snap_id=None,
                   notes="") -> int

# --- constituent properties (in-situ values) ---
save_constituent_property(constituent_type, constituent_id, property_name,
                           value, unit, source_tag,
                           print_config_id=None,
                           inference_run_id=None, notes="") -> int
get_constituent_properties(constituent_type, constituent_id,
                            print_config_id=None) -> dict
# returns: {property_name: {value, unit, source_tag, inference_run_id, created_at}}
# lookup order: print_config_id-specific first, then global (NULL)

# --- composite properties ---
save_composite_property(print_config_id, property_name, value, unit,
                         source_tag, inference_run_id=None,
                         measurement_id=None, temperature_C=None) -> int
get_composite_properties(print_config_id,
                          preferred_source=None) -> dict

# --- value resolution ---
get_canonical_value(print_config_id, property_name,
                    constituent_type=None, constituent_id=None)
    -> (value, unit, source_tag, inference_run_id or None)
# Resolution order:
#   1. property_preferences override
#   2. inputted (current session)
#   3. inferred — most recent from constituent_property_values
#   4. web — fibers.neat_* / polymers.neat_*
#   5. predicted

# --- property preferences ---
set_property_preference(print_config_id, property_name, preferred_source)

# --- material card (full view) ---
get_material_card(print_config_id) -> {
    "print_config":       dict,
    "fiber":              dict,    # neat properties from fibers table
    "polymer":            dict,    # neat properties from polymers table
    "printer":            dict,
    "microstructure":     dict,    # latest snapshot with provenance_json parsed
    "constituent_props":  dict,    # all with provenance, best per property
    "composite_props":    dict,    # all with provenance
    "experimental":       list,    # raw measurements
    "inference_history":  list,    # all runs newest first
}
```

---

## 10. GUI Integration

### Forward GUI (`gui.py`)
- Add **print config selector** (fiber + polymer + printer dropdowns) replacing the current separate fiber/polymer dropdowns
- Each auto-filled field shows its source tag: `"3200 MPa  [inferred, Stage 1, 2026-03-15]"`
- **"Save prediction to card"** button after a successful forward prediction

### Inverse GUI (`gui_inverse.py`)
- Add **print config selector** at top
- **"Load microstructure from card"** — pulls latest snapshot and sets those fields to Fixed
- **"Load constituent properties from card"** — pulls inferred props as initial values / fixed inputs
- After solve: **"Save to card"** dialog — commits in one transaction:
  - `inference_run` record
  - `microstructure_snapshot` (if microstructure was free, with correct provenance_json)
  - `constituent_property_values` for each inferred constituent prop
  - `composite_property_values` for forward predictions at inferred params

### Thermal Inverse GUI (`gui_thermal_inverse.py`)
- Same print config selector
- After solve: saves inferred constituent k parameters to `constituent_property_values`
- Saves per-temperature composite k predictions to `composite_property_values`

### Material Card Viewer (`gui_material_card.py` — new file)
Standalone window, launchable from any GUI.

**Panels:**
1. **Header** — print config selector, fiber/polymer/printer names
2. **Constituent properties** — side-by-side table: web baseline vs. in-situ inferred, source badge per row, click to see which inference run produced it
3. **Microstructure** — latest snapshot with per-field provenance, full history collapsible
4. **Composite properties** — predicted vs. experimental, toggle per row (sets `property_preferences`), temperature filter for thermal/CTE data
5. **Inference history** — timeline of all runs, click to expand full `inputs_json` / `outputs_json`
6. **Actions**: "Use for forward prediction", "Use for inverse setup", "Export card as JSON/CSV"

---

## 11. Open Decisions

| # | Question | Options |
|---|---|---|
| 1 | **Migration vs. clean break** — current tables have data | (a) Migrate existing rows into new schema, (b) clean `--reset` |
| 2 | **Constituent property scope** — when Stage 2 infers fiber_CTE on Printer A, is it auto-available for Printer B? | (a) Store as `print_config_id=NULL` (global — recommended), (b) keep scoped, require explicit "promote to global" step |
| 3 | **Temperature-dependent CTE** — is CTE measured at multiple temperatures like k vs. T? | Determines whether `composite_property_values.temperature_C` is used for CTE rows |
| 4 | **Experimental data entry UI** — dedicated panel or inline in inverse GUI? | (a) separate "Enter measurements" tab in material card viewer, (b) entered inline as target values in inverse GUI (current pattern) |

---

## 12. Implementation Phases

| Phase | Files | Description |
|---|---|---|
| **1** | `init_db.py` | Schema v2: add 7 new tables, extend fibers/polymers. Keep old tables during transition. |
| **2** | `db.py` | Implement all new API functions from Section 9. |
| **3** | `gui_inverse.py` | Print config selector + "Save to card" transaction after solve. |
| **4** | `gui.py` | Replace fiber/polymer dropdowns with print config selector. Show provenance per field. Add "Save prediction" button. |
| **5** | `gui_material_card.py` | Build material card viewer (new file). |
| **6** | `gui_thermal_inverse.py` | Wire up save-to-card for inferred constituent k values. |
| **7** | `gui_material_card.py` | Cross-printer wizard: "Create new card using existing constituent properties". |

---

## 13. Current Schema Gaps (Summary)

| Gap | Impact | Fixed in Phase |
|---|---|---|
| No CTE / k columns on fibers/polymers | Thermal and thermoelastic stages can't store constituent baselines | 1 |
| No `print_configs` table | Can't distinguish same material on different printers | 1 |
| `inferred_properties` uses flat columns (E1,E2,CTE1...) | Can't store arbitrary property sets; breaks for new surrogate outputs | 1 |
| No `experimental_measurements` table | Experimental targets are not persisted, only used and discarded | 1 |
| No `uncertainty` column on experimental data | σ values typed in GUI are lost after each solve | 1 |
| No provenance tags on any stored value | Can't tell if a value is inferred, predicted, or from a datasheet | 1–2 |
| `microstructure` has no version history | Updating orientation silently overwrites previous values | 1 |
| `microstructure` has single source_tag per row | Can't record that mf was inputted but a11 was inferred in the same snapshot | 1 |
| No `property_preferences` | No way to express "use experimental k11, but predicted CTE22" | 1–2 |
| Material card is fiber+polymer only | No printer differentiation; same material on two printers is one card | 1 |
| `in_situ: null` fields in fibers.json / polymers.json | Misleading leftover — implies in-situ values belong in the JSON file | Remove from JSON files |
