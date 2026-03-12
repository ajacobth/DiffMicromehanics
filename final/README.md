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

# Launch the Inverse GUI  (find microstructure that hits target properties)
python gui_inverse.py
```

See [SETUP_AND_RUN.md](../SETUP_AND_RUN.md) for full details.
