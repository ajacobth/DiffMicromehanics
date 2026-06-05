#!/usr/bin/env bash
# First-time setup for DiffMicromechanics / MateriAl agent.
# Run once from the final/ directory:
#
#   bash setup.sh
#
# What this does:
#   1. Checks you are in the right folder
#   2. Checks the conda environment is active
#   3. Installs Python packages from requirements.txt
#   4. Creates the uploads folder for thermal CSV files
#   5. Initialises the SQLite database and seeds the material library
#   6. Verifies the surrogate models load correctly

set -e  # stop on first error

# ── 1. Check we are in final/ ─────────────────────────────────────────────────
if [ ! -f "app_chat.py" ] || [ ! -d "db" ]; then
    echo "ERROR: Run this script from the final/ directory."
    echo "  cd /path/to/DiffMicromehanics/final"
    echo "  bash setup.sh"
    exit 1
fi

echo ""
echo "=== DiffMicromechanics — First-time setup ==="
echo ""

# ── 2. Check conda environment ────────────────────────────────────────────────
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "WARNING: No conda environment is active."
    echo "  Run: conda activate jax_trial"
    echo "  Then re-run: bash setup.sh"
    exit 1
fi

echo "Active environment: $CONDA_DEFAULT_ENV"

if [ "$CONDA_DEFAULT_ENV" != "jax_trial" ]; then
    echo "WARNING: Expected environment 'jax_trial' but found '$CONDA_DEFAULT_ENV'."
    echo "  Run: conda activate jax_trial"
    read -p "  Continue anyway? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        exit 1
    fi
fi

# ── 3. Install Python packages ────────────────────────────────────────────────
echo ""
echo "--- Installing packages from requirements.txt ---"
pip install -r requirements.txt --quiet
echo "    Done."

# ── 4. Create uploads folder ──────────────────────────────────────────────────
echo ""
echo "--- Creating data/uploads/ folder ---"
mkdir -p data/uploads
echo "    Done. Place thermal K vs T CSV files here before running thermal inverse."

# ── 5. Initialise the database ────────────────────────────────────────────────
echo ""
echo "--- Initialising database ---"
if [ -f "data/micromechanics.db" ]; then
    echo "    data/micromechanics.db already exists."
    read -p "    Re-initialise? This will DELETE all saved cards and results. [y/N] " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        rm data/micromechanics.db
        python db/init_db.py
    else
        echo "    Skipped — keeping existing database."
    fi
else
    python db/init_db.py
fi

# ── 6. Verify surrogate models ────────────────────────────────────────────────
echo ""
echo "--- Verifying surrogate models ---"
python scripts/test_setup.py

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  To run the chat agent:"
echo "    conda activate jax_trial"
echo "    streamlit run app_chat.py"
echo ""
echo "  To run the GUI:"
echo "    conda activate jax_trial"
echo "    python gui.py"
echo "============================================================"
echo ""
