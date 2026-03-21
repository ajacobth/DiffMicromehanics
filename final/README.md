# Setup and Run Instructions

> **New here? Start with the setup guide before running anything.**
>
> **[SETUP_AND_RUN.md](../SETUP_AND_RUN.md)**
>
> It covers: installing Conda, creating a Python environment, installing all
> required packages (Windows / Mac / Linux, CPU and GPU), running the test
> script to verify your setup, and step-by-step instructions for both GUIs.

---

## Quick reference (once your environment is set up)

```bash
# Activate your environment first (every new terminal session)
conda activate diffmech

# Navigate here
cd path/to/DiffMicromehanics/final

# Verify everything still works
python test_setup.py

# Launch the Forward GUI  (predict properties from microstructure inputs)
python gui.py

# Launch the Inverse GUI  (elastic / thermoelastic inverse design)
python gui_inverse.py

# Launch the Thermal Inverse GUI  (recover constituent thermal conductivities)
python gui_thermal_inverse.py

# Run the Thermal Inverse from the command line (no GUI)
python run_inverse_thermal.py                            # uses thermal_problem.json
python run_inverse_thermal.py --problem my_problem.json
python run_inverse_thermal.py --problem p.json --output_dir results/
```

The thermal inverse GUI can also be opened from within `gui_inverse.py` via
the **Thermal Inverse** button at the top of that window.

See [SETUP_AND_RUN.md](../SETUP_AND_RUN.md) for full details, including the
new Section 11 (thermal inverse estimation).
