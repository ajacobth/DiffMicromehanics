# Evaluation Plan — DiffMicromechanics
## Step-by-step manual testing guide

Work through each section in order. Each test lists what to do, what to look for, and what a failure looks like. DB verification steps use your SQLite viewer pointed at `data/micromechanics.db`.

---

## Section 0 — Environment sanity

**0.1** — Activate the environment and run the test script:
```bash
conda activate diffmech
cd path/to/DiffMicromehanics/final
python test_setup.py
```
- **Pass:** no errors, all imports succeed, surrogate models load
- **Fail:** missing packages, JAX errors, model checkpoint not found

**0.2** — Verify the data folder has the seed files:
```bash
ls data/
```
- Should contain: `fibers.json`, `polymers.json`
- Open `fibers.json` and confirm: E1 for Carbon Fiber T300 = **230000** (MPa, not 230 GPa — units in the JSON are already in MPa)

---

## Section 1 — Database setup (`init_db.py`)

### 1.1 — Normal first-run
```bash
python init_db.py
```
- **Pass:** prints "Schema ready", "seeded 3 fiber(s)", "seeded 2 polymer(s)", "Done."
- **Fail:** any exception, wrong counts

### 1.2 — DB viewer check after init
Open `data/micromechanics.db` in your SQLite viewer. Run:
```sql
SELECT name, neat_E1, neat_CTE1, neat_k1 FROM fibers;
```
Expected:

| name | neat_E1 | neat_CTE1 | neat_k1 |
|---|---|---|---|
| Carbon Fiber T300 | 230000 | -0.7 | 10.5 |
| E-Glass | 72000 | 5.0 | 1.05 |
| AS4 | 240000 | -0.4 | 6.83 |

```sql
SELECT name, neat_E1, neat_CTE, neat_k FROM polymers;
```
Expected: PESU Ultrason (3600, 55.0, 0.26), Epoxy 3501-6 (4200, 60.0, 0.17)

### 1.3 — Re-run without --reset (should prompt)
```bash
python init_db.py
```
- Should ask: `data/micromechanics.db already exists. Reset and re-seed? [y/N]`
- Type `N` → nothing changes
- **Verify:** fiber count is still 3 (not 6 from double-seeding)
```sql
SELECT COUNT(*) FROM fibers;   -- must be 3, not 6
SELECT COUNT(*) FROM polymers; -- must be 2, not 4
```

### 1.4 — Re-run with --reset flag
```bash
python init_db.py --reset
```
- Should delete the DB, recreate, and reseed
- **Verify:** fiber count is 3 again, IDs restart at 1

### 1.5 — Edge case: missing fibers.json
```bash
mv data/fibers.json data/fibers.json.bak
python init_db.py --reset
```
- **Pass:** prints `[skip] fibers.json not found`, completes without crash, 0 fibers in DB
- Restore: `mv data/fibers.json.bak data/fibers.json` then `python init_db.py --reset`

### 1.6 — Edge case: corrupted JSON
```bash
echo "this is not json" > data/fibers.json
python init_db.py --reset
```
- **Pass:** crashes with a clear JSON parse error (not a silent wrong state)
- Restore: `git checkout data/fibers.json` then `python init_db.py --reset`

---

## Section 2 — Forward GUI (`gui.py`)

### 2.1 — Launch and DB status
```bash
python gui.py
```
- The "Material Library" bar should show: **"Library: 3 fibers, 2 polymers"**
- If DB doesn't exist or wasn't init'd, a red warning should appear and dropdowns should be disabled

### 2.2 — Load elastic model
- Select "elastic" from the model dropdown, click **Load Model**
- Input fields should appear for: e1, e2, g12, f_nu12, f_nu23, ar, fiber_massfrac, fiber_density, matrix_modulus, matrix_poisson, matrix_density, a11, a22, a12, a13, a23

### 2.3 — Fiber auto-fill
- Select **Carbon Fiber T300** from the Fiber dropdown
- **Expected fields filled:**
  - e1 → 230000 (MPa)
  - e2 → 15000
  - g12 → 15000
  - f_nu12 → 0.20
  - fiber_density → 1760
- **Pass:** values appear immediately with no button press needed

### 2.4 — Polymer auto-fill
- Select **PESU Ultrason** from the Polymer dropdown
- **Expected fields filled:**
  - matrix_modulus → 3600
  - matrix_poisson → 0.37
  - matrix_density → 1370

### 2.5 — Manual override persists
- With Carbon Fiber T300 selected, manually change `e1` to 999
- Select a different fiber (E-Glass), then select Carbon Fiber T300 again
- **Pass:** e1 should revert to 230000 (auto-fill re-applies on re-select)
- **Fail to watch for:** field keeps 999 after re-selecting same fiber

### 2.6 — Select "— none —" does not clear fields
- Fill some fields by selecting a fiber
- Change Fiber dropdown to "— none —"
- **Pass:** fields stay at their previous values (do not blank out)

### 2.7 — Run a prediction
- Load elastic model, fill all inputs with reasonable values (use auto-fill for fiber + polymer, then set ar=20, fiber_massfrac=0.3, a11=0.6, a22=0.2, a12=a13=a23=0.0)
- Click **Predict**
- **Pass:** output panel shows E1, E2, E3, G12, G13, G23, nu12, nu13, nu23 in GPa
- Values should be physically reasonable (E1 between 10–150 GPa for CF composites)

### 2.8 — Switch model without crashing
- With elastic loaded and fields filled, switch to "thermoelastic"
- Click **Load Model**
- **Pass:** input fields update (CTE fields appear), previous field values that exist in both models are retained or cleared gracefully

### 2.9 — Switch to thermal model
- Load "thermal" model
- Fiber auto-fill should populate the same mechanical + density fields
- Run Predict
- **Pass:** outputs are k11, k22, k33 in W/m·K

### 2.10 — In-situ toggle (initial state)
- The "In-situ" radio button should be **greyed out** (no inference data exists yet)
- Try clicking it anyway
- **Pass:** does nothing / stays on "Neat"

### 2.11 — Missing DB while GUI is running
- With gui.py open, rename `data/micromechanics.db` to something else in your file manager
- Try loading a model or selecting a material
- **Pass:** graceful error message, no crash
- Restore the DB file

---

## Section 3 — Inverse GUI (`gui_inverse.py`)

### 3.1 — Launch
```bash
python gui_inverse.py
```
- Should open with elastic model options visible

### 3.2 — Load elastic model
- Click **Load Model** with "elastic" selected
- Input rows appear, each with a value entry and a Fixed/Free toggle
- All fields default to **Fixed**

### 3.3 — Basic elastic solve (minimal free variables)
Set up as follows:
- Load AS4 fiber via auto-fill (if available in dropdowns — if not, enter manually: e1=240000, e2=14000, g12=28000, f_nu12=0.2, f_nu23=0.28, fiber_density=1780)
- Set matrix_density = 1370, fiber_massfrac = 0.3 → Fixed
- Set **Free**: matrix_modulus, matrix_poisson, a11, a22
- Bounds: matrix_modulus [2000, 5000], matrix_poisson [0.30, 0.45], a11 [0.2, 0.9], a22 [0.05, 0.3]
- In Targets panel: check E1=15000 MPa (σ=0), E2=5000 MPa (σ=0), G12=2500 MPa (σ=0)
- Click **SOLVE**
- **Pass:** solver runs, converges (loss < 0.1), result panel shows inferred values
- **Watch for:** thread hangs, GUI freezes permanently (brief freeze during solve is normal)

### 3.4 — Edge case: zero free variables
- Set all fields to Fixed, click SOLVE
- **Pass:** error message or immediate return with loss=0 (nothing to optimise)
- **Fail:** crash or silent hang

### 3.5 — Edge case: all free variables
- Set all fields to Free, enter physically impossible targets (E1=500000 MPa)
- Click SOLVE
- **Pass:** solver completes without crash, loss is large, result shows clearly bad values
- **Fail:** infinite loop, crash, GUI freeze

### 3.6 — Edge case: non-numeric target entry
- In the Targets panel, type `"abc"` into the E1 target field
- Click SOLVE
- **Pass:** validation error shown, solve does not start
- **Fail:** crash or Python exception in terminal

### 3.7 — Thermoelastic solve
- Switch to "thermoelastic", Load Model
- Fill fiber and polymer mechanical properties as Fixed
- Set a11=0.6, a22=0.2, a12=a13=a23=0 as Fixed
- Set ar=20, fiber_massfrac=0.3 as Fixed
- Set **Free**: f_CTE1, f_CTE2, matrix_CTE
- Targets: CTE11 = 2e-6 (1/K), CTE22 = 30e-6 (1/K)
- Click SOLVE
- **Pass:** inferred CTE values are physically plausible (fiber CTE axial near -0.5e-6 to 0, matrix CTE near 40-60e-6)

### 3.8 — Open Thermal Inverse from this GUI
- Click **"Open Thermal Inverse Solver"** (or "Thermal Inverse" button)
- **Pass:** a new window opens (`ThermalInverseWindow`)
- **Fail:** button missing, window crashes on open

---

## Section 4 — Thermal Inverse GUI (`gui_thermal_inverse.py`)

### 4.1 — Launch standalone
```bash
python gui_thermal_inverse.py
```
- Window should open with fields for Vf, aspect ratio, orientation inputs, and a CSV load button

### 4.2 — Load a CSV with correct format
The solver expects columns for temperature and composite conductivities. Create a test CSV:
```
T_C,K11,K22,K33
25,2.1,0.5,0.5
50,2.0,0.48,0.48
75,1.9,0.46,0.46
100,1.85,0.45,0.45
```
Save as `test_thermal.csv` in the `final/` folder, then load it in the GUI.
- **Pass:** file loads, data visible in the GUI, temperature range shown

### 4.3 — Run estimation
- Set: Vf=0.174, ar=20, a11=0.77, a22=0.16, a12=a13=a23=0
- Load the CSV above
- Click **Run** (or Estimate)
- **Pass:** three live plots appear: (1) composite K predictions vs. data, (2) fiber k vs. T, (3) matrix k vs. T
- **Watch for:** plot window not updating, estimation running on GUI thread (should be threaded)

### 4.4 — Edge case: CSV with wrong column name
Create a CSV with column `Temp` instead of `T_C`:
```
Temp,K11,K22
25,2.1,0.5
```
- **Pass:** error shown about missing temperature column, no crash
- **Fail:** KeyError traceback, silent wrong result, or hang

### 4.5 — Edge case: CSV with only 1 row of data
```
T_C,K11,K22,K33
25,2.1,0.5,0.5
```
- **Pass:** solver runs (may converge poorly), no crash
- **Fail:** division by zero, shape error, crash

### 4.6 — Edge case: CSV with NaN values
```
T_C,K11,K22,K33
25,2.1,,0.5
50,nan,0.48,0.48
```
- **Pass:** graceful handling (skip NaN rows, or error message)
- **Fail:** silent wrong result passed to solver

### 4.7 — Close window mid-estimation
- Start an estimation (click Run)
- Immediately close the window while solver is running
- **Pass:** no crash in terminal, background thread finishes or is killed cleanly
- **Fail:** Python exception, zombie thread printing to terminal after window close

---

## Section 5 — CLI thermal inverse (`run_inverse_thermal.py`)

### 5.1 — Normal run
```bash
python run_inverse_thermal.py
```
- **Pass:** prints progress, writes results to current directory
- **Fail:** FileNotFoundError for `measurements.csv` (the default problem.json references this file)

> **Note:** `thermal_problem.json` references `data: "measurements.csv"` which likely doesn't exist yet. The CLI will fail until you either:
> (a) create a `measurements.csv` in the right place, or
> (b) pass a problem file pointing to your test CSV

### 5.2 — Run with a custom problem file
```bash
python run_inverse_thermal.py --problem thermal_problem.json
```
Same as above — verify it reads the right file.

### 5.3 — Run with custom output dir
```bash
mkdir -p /tmp/thermal_test
python run_inverse_thermal.py --problem thermal_problem.json --output_dir /tmp/thermal_test
```
- **Pass:** results written to `/tmp/thermal_test/`
- **Fail:** results still go to current directory, output_dir flag ignored

### 5.4 — Edge case: non-existent output dir
```bash
python run_inverse_thermal.py --output_dir /tmp/does_not_exist_xyz
```
- **Pass:** creates the directory, or shows clear error
- **Fail:** crash without mkdir

### 5.5 — Edge case: malformed problem JSON
```bash
echo '{"data": "measurements.csv", "n_restarts": "ten"}' > bad_problem.json
python run_inverse_thermal.py --problem bad_problem.json
```
- **Pass:** type error caught cleanly with a helpful message
- **Fail:** silent wrong cast, or deep traceback from inside the solver

### 5.6 — n_restarts = 0
Edit a copy of `thermal_problem.json` with `"n_restarts": 0`, run it.
- **Pass:** either completes with a single run or shows a clear error
- **Fail:** infinite loop, crash

---

## Section 6 — Database integrity after GUI operations

After running a solve in gui_inverse.py (Section 3.3), open the DB viewer and run these checks.

> **Note:** Sections 5 and 6 of the build plan (Save to Card, Material Card Viewer) are marked **pending**. Until those are implemented, the inference GUIs do not write to the DB automatically. These DB checks apply once Save to Card is implemented, or can be tested by calling `db.py` functions directly in a Python shell.

### 6.1 — Manually write a test record via Python
```python
import db
# Create a print config
cfg_id = db.create_print_config("Test CF-PESU", fiber_id=1, polymer_id=1)

# Save a microstructure snapshot
snap_id = db.save_microstructure_snapshot(
    cfg_id, mf=0.3, ar=20.0,
    a11=0.6, a22=0.2, a12=0.0, a13=0.0, a23=0.0,
    provenance={"mf": "inputted", "ar": "inputted", "a11": "inferred", "a22": "inferred"}
)

# Save an inference run
run_id = db.save_inference_run(
    print_config_id=cfg_id,
    stage="elastic",
    inputs={"e1": 230000, "matrix_modulus": 3600},
    outputs={"E1": 15000, "E2": 5000},
    solver_cfg={"method": "lbfgs", "maxiter": 300},
    loss=0.0042,
    microstructure_snap_id=snap_id,
)

# Save a constituent property
db.save_constituent_property(
    constituent_type="polymer",
    constituent_id=1,
    property_name="matrix_modulus",
    value=3150.0,
    source_tag="inferred",
    unit="MPa",
    inference_run_id=run_id,
)

# Save an experimental measurement
db.save_experimental_measurement(
    print_config_id=cfg_id,
    property_name="E1",
    value=15000,
    unit="MPa",
    uncertainty=300,
)
print("All writes succeeded, run_id =", run_id)
```
- **Pass:** no exceptions, run_id printed

### 6.2 — DB viewer checks after 6.1
```sql
-- Should be 1 print_config
SELECT * FROM print_configs;

-- Should be 1 microstructure snapshot with provenance JSON
SELECT id, mf, ar, a11, a22, provenance_json FROM microstructure_snapshots;

-- Provenance JSON should have a11="inferred", mf="inputted"
-- Parse it manually or look at the raw column

-- Should be 1 inference run with stage='elastic', loss=0.0042
SELECT id, stage, loss FROM inference_runs;

-- Should be 1 constituent property (matrix_modulus = 3150)
SELECT constituent_type, property_name, value, source_tag FROM constituent_property_values;

-- Should be 1 experimental measurement (E1 = 15000)
SELECT property_name, value, uncertainty FROM experimental_measurements;
```

### 6.3 — Foreign key integrity
```sql
-- All microstructure snapshots must have a valid print_config_id
SELECT ms.id, ms.print_config_id, pc.name
FROM microstructure_snapshots ms
LEFT JOIN print_configs pc ON pc.id = ms.print_config_id
WHERE pc.id IS NULL;
-- Expected: 0 rows

-- All inference_runs must have valid print_config_id
SELECT ir.id FROM inference_runs ir
LEFT JOIN print_configs pc ON pc.id = ir.print_config_id
WHERE pc.id IS NULL;
-- Expected: 0 rows
```

### 6.4 — get_print_config_card round-trip
```python
import db, json
card = db.get_print_config_card(1)
print(json.dumps({k: str(v)[:80] for k, v in card.items()}, indent=2))
```
- **Pass:** returns dict with keys: config, fiber, polymer, printer, microstructure, inference_runs, constituent_properties, experimental_measurements, composite_properties
- **Fail:** KeyError, None where dict expected, foreign key resolve fails

### 6.5 — get_canonical_value resolution
```python
import db
# Should return the experimental measurement (priority over predicted)
db.save_composite_property(1, "E1", 14800, "predicted", "MPa")
val = db.get_canonical_value(1, "E1")
print(val)  # Should return the experimental one (15000), not predicted (14800)
```
- **Pass:** returns `source_tag='experimental'`, value=15000
- **Fail:** returns predicted value (wrong priority order)

### 6.6 — Global vs. card-scoped constituent properties
```python
import db
# Write a global property (print_config_id=None)
db.save_constituent_property("polymer", 1, "k_m", 0.28, "inferred", "W/m·K", print_config_id=None)

# Write a card-scoped property
db.save_constituent_property("polymer", 1, "k_m", 0.31, "inferred", "W/m·K", print_config_id=1)

# Query with include_global=True for card 1 — should get BOTH rows
rows = db.get_constituent_properties("polymer", 1, "k_m", print_config_id=1, include_global=True)
print(len(rows))  # Should be 2

# Query with include_global=False — should get only the card-scoped one
rows2 = db.get_constituent_properties("polymer", 1, "k_m", print_config_id=1, include_global=False)
print(len(rows2))  # Should be 1
```

### 6.7 — Double-seed guard
Re-run `python init_db.py` and answer `y` to reset, then re-run 6.1's Python writes:
```sql
SELECT COUNT(*) FROM fibers;    -- must still be 3
SELECT COUNT(*) FROM polymers;  -- must still be 2
```
After --reset, all the material card tables should be empty too:
```sql
SELECT COUNT(*) FROM print_configs;  -- 0
SELECT COUNT(*) FROM inference_runs; -- 0
```

---

## Section 7 — Thermal inverse helper functions

### 7.1 — save_thermal_inverse_results
```python
import db
cfg_id = db.create_print_config("Thermal Test", fiber_id=1, polymer_id=1)
run_id = db.save_thermal_inverse_results(
    print_config_id=cfg_id,
    fiber_id=1,
    polymer_id=1,
    parametric_outputs={"k_f1": 10.5, "k_f2": 0.9, "k_m": 0.27, "p1": 0.5, "p2": 0.3},
    solver_cfg={"n_restarts": 10},
    loss=0.0012,
)
print("run_id =", run_id)
```

### 7.2 — Verify thermal results in DB viewer
```sql
-- Should show k_f1 and k_f2 stored globally (print_config_id IS NULL)
SELECT constituent_type, property_name, value, print_config_id, source_tag
FROM constituent_property_values
WHERE inference_run_id = 2;  -- use the run_id from above

-- k_f1 and k_f2 → print_config_id = NULL (global)
-- k_m          → print_config_id = NULL (global)
-- thermal_p1   → print_config_id = <cfg_id> (card-scoped)
-- thermal_p2   → print_config_id = <cfg_id> (card-scoped)
```

### 7.3 — get_thermal_constituent_inputs fallback chain
```python
import db
# Fresh DB with no inferred k values — should fall back to neat values from fibers table
result = db.get_thermal_constituent_inputs(999, fiber_id=1, polymer_id=1)
# print_config_id=999 doesn't exist — should not crash, should return neat values
print(result)
# Expected: k_f1=10.5, k_f2=0.96, k_m=0.26
```

---

## Section 8 — Deliberate stress/break tests

These are things that should either fail gracefully or expose bugs.

### 8.1 — Negative fiber volume fraction
In any GUI, set fiber_massfrac = -0.1, run predict.
- **Pass:** validation error, or physically nonsensical but non-crashing output
- **Fail:** surrogate returns NaN or explodes, no error to user

### 8.2 — Orientation tensor components summing to > 1
Set a11=0.9, a22=0.9, run elastic forward.
- a11 + a22 should not exceed 1 (orientation tensor constraint). The surrogate was trained on valid tensors.
- **Watch for:** prediction outside any physically meaningful range. No validation needed necessarily, but note the output.

### 8.3 — Very high or low Vf
Set fiber_massfrac = 0.99 (near pure fiber), run thermal forward.
- **Pass:** model outputs something, no crash
- Note whether output is physically unreasonable (k11 ≈ fiber k1 ≈ 10.5 would be expected at Vf→1)

### 8.4 — Empty string in a numeric field
In the forward GUI, clear the `e1` field completely and click Predict.
- **Pass:** validation error or safe fallback
- **Fail:** ValueError crash, empty string passed to numpy

### 8.5 — Concurrent GUI instances
Open `gui.py` and `gui_inverse.py` at the same time.
- Both access the same SQLite DB
- **Pass:** both run, no DB lock errors (WAL mode should handle this)
- **Fail:** `database is locked` error

### 8.6 — Run init_db.py --reset while a GUI is open
- Open `gui.py`, leave it running
- In a terminal: `python init_db.py --reset`
- **Pass:** reset succeeds; GUI may show stale data but should not crash on next DB access
- **Fail:** `database is locked`, reset hangs

### 8.7 — DB file permissions
```bash
chmod 444 data/micromechanics.db   # make read-only
python gui.py
```
- Try selecting a material (read-only, should work)
- Try running a solve and saving (if Save to Card exists) — should show permission error
- Restore: `chmod 644 data/micromechanics.db`

### 8.8 — Save same print_config name twice
```python
import db
id1 = db.create_print_config("Duplicate", fiber_id=1, polymer_id=1)
id2 = db.create_print_config("Duplicate", fiber_id=1, polymer_id=1)
print(id1, id2)  # Both succeed — name is NOT unique-constrained
```
- **Check:** does the schema have a UNIQUE constraint on `print_configs.name`? (It does not.)
- This means duplicate card names are silently allowed. Note this as a potential UX issue.

### 8.9 — constituent_property_values for a non-existent fiber_id
```python
import db
# fiber_id=999 does not exist — foreign key should reject this
try:
    db.save_constituent_property("fiber", 999, "E1", 230000, "inferred", "MPa")
    print("UNEXPECTED: no FK error raised")
except Exception as e:
    print("Expected FK error:", e)
```
- **Note:** SQLite foreign key enforcement requires `PRAGMA foreign_keys=ON`, which `_connect()` does set. Verify the error is raised.

### 8.10 — import_thermal_csv with missing file
```python
import db
cfg_id = 1  # from earlier
try:
    db.import_thermal_csv(cfg_id, "/nonexistent/path.csv")
except FileNotFoundError as e:
    print("Expected:", e)
except Exception as e:
    print("Unexpected error type:", type(e), e)
```

---

## Section 9 — Things that are NOT yet built (known pending)

These are expected to be missing. Document the current state rather than treating as failures.

| Feature | Expected state |
|---|---|
| "Save to Card" button in `gui_inverse.py` after a solve | Not yet implemented (Step 5) |
| "Load from Card" populating inverse GUI fields | Not yet implemented (Step 5) |
| Material Card Viewer (`gui_material_card.py`) | Shell file exists, full UI pending (Step 6) |
| In-situ toggle becoming active after first inverse solve | Depends on Step 5 |
| Thermal inverse Save to Card | Not yet implemented (Phase 6) |

---

## Summary checklist

| Section | What it tests | Done? |
|---|---|---|
| 0 | Environment, imports, model checkpoints | |
| 1 | init_db.py — normal, re-run, missing file, corrupt JSON | |
| 2 | Forward GUI — model loading, auto-fill, prediction, edge cases | |
| 3 | Inverse GUI — elastic solve, thermoelastic, edge cases | |
| 4 | Thermal inverse GUI — CSV load, estimation, edge cases | |
| 5 | CLI thermal inverse — normal, bad inputs, missing files | |
| 6 | DB integrity — Python API, FK checks, canonical value resolution | |
| 7 | Thermal helpers — save/read thermal inverse results | |
| 8 | Stress tests — bad inputs, concurrency, permissions, duplicates | |
| 9 | Pending features — confirm absent but not crashing | |
