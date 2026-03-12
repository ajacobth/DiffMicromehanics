"""test_setup.py – Run this script to verify your environment is correctly set up.

Usage
-----
    python test_setup.py

All checks should print PASS. If any prints FAIL, follow the instructions shown.
"""

import sys
import os

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"

errors_found = False

def check(label, condition, hint=""):
    global errors_found
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}")
        if hint:
            print(f"         --> {hint}")
        errors_found = True

# ── 1. Python version ─────────────────────────────────────────────────────────
print("\n=== 1. Python version ===")
major, minor = sys.version_info[:2]
print(f"  {INFO}  Python {major}.{minor}.{sys.version_info.micro}")
check(
    "Python >= 3.9",
    major == 3 and minor >= 9,
    "Install Python 3.10 via conda: conda create -n diffmech python=3.10"
)

# ── 2. Core numerical packages ────────────────────────────────────────────────
print("\n=== 2. Core packages ===")

try:
    import numpy as np
    check(f"numpy {np.__version__}", True)
except ImportError:
    check("numpy", False, "pip install numpy==1.26.2")

try:
    import scipy
    check(f"scipy {scipy.__version__}", True)
except ImportError:
    check("scipy", False, "pip install scipy==1.11.4")

try:
    import matplotlib
    check(f"matplotlib {matplotlib.__version__}", True)
except ImportError:
    check("matplotlib", False, "pip install matplotlib==3.8.2")

# ── 3. JAX ────────────────────────────────────────────────────────────────────
print("\n=== 3. JAX ===")

try:
    import jax
    import jax.numpy as jnp
    check(f"jax {jax.__version__}", True)

    # Test basic computation
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    check("jax basic computation (sum)", float(y) == 6.0)

    # Check platform (CPU expected for no-GPU systems)
    backend = jax.default_backend()
    print(f"  {INFO}  JAX backend: {backend}")
    if backend == "cpu":
        print(f"  {INFO}  Running on CPU (normal if you have no GPU)")
    elif backend in ("gpu", "metal"):
        print(f"  {INFO}  Running on {'GPU (CUDA)' if backend == 'gpu' else 'GPU (Apple Metal)'} -- great!")

    # Test JIT compilation
    @jax.jit
    def simple_fn(x):
        return jnp.dot(x, x)

    result = simple_fn(jnp.ones(10)).block_until_ready()
    check("jax JIT compilation", float(result) == 10.0)

    # Test grad
    grad_fn = jax.grad(lambda x: jnp.sum(x ** 2))
    g = grad_fn(jnp.array([1.0, 2.0, 3.0]))
    check("jax autodiff (grad)", list(g) == [2.0, 4.0, 6.0])

except ImportError:
    check("jax", False,
          "Windows/Linux CPU: pip install 'jax[cpu]==0.4.26'\n"
          "         Mac Apple Silicon GPU: pip install jax-metal==0.1.0")

# ── 4. ML packages ────────────────────────────────────────────────────────────
print("\n=== 4. ML packages ===")

try:
    import flax
    check(f"flax {flax.__version__}", True)
except ImportError:
    check("flax", False, "pip install flax==0.8.3")

try:
    import optax
    check(f"optax {optax.__version__}", True)
except ImportError:
    check("optax", False, "pip install optax==0.1.7")

try:
    import jaxopt
    check(f"jaxopt {jaxopt.__version__}", True)
except ImportError:
    check("jaxopt", False, "pip install jaxopt==0.8.5")

try:
    import chex
    check(f"chex {chex.__version__}", True)
except ImportError:
    check("chex", False, "pip install chex==0.1.85")

try:
    import ml_collections
    check("ml-collections", True)
except ImportError:
    check("ml-collections", False, "pip install ml-collections==0.1.1")

try:
    import orbax.checkpoint
    check("orbax-checkpoint", True)
except ImportError:
    check("orbax-checkpoint", False, "pip install orbax-checkpoint==0.4.8")

# ── 5. GUI (tkinter) ──────────────────────────────────────────────────────────
print("\n=== 5. GUI (tkinter) ===")
try:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    root.destroy()
    check("tkinter (GUI toolkit)", True)
except Exception as e:
    check("tkinter", False,
          "On Linux: sudo apt-get install python3-tk\n"
          "         On Windows/Mac: reinstall Python from conda (tkinter is included)")

# ── 6. Folder structure ───────────────────────────────────────────────────────
print("\n=== 6. Folder structure ===")

_HERE = os.path.dirname(os.path.abspath(__file__))

check(
    "final/ folder found",
    os.path.isfile(os.path.join(_HERE, "gui.py")),
    "Make sure you are running this script from the final/ folder"
)

el_dir = os.path.normpath(os.path.join(_HERE, "..", "EL_surrogate"))
check(
    "EL_surrogate/ folder found (sibling of final/)",
    os.path.isdir(el_dir),
    f"Expected at: {el_dir}\n"
    "         Make sure you cloned the FULL repository, not just the final/ folder"
)

check(
    "models/elastic/ exists",
    os.path.isdir(os.path.join(_HERE, "models", "elastic")),
    "The models/elastic/ folder is missing. Contact the project maintainer."
)
check(
    "models/thermoelastic/ exists",
    os.path.isdir(os.path.join(_HERE, "models", "thermoelastic")),
    "The models/thermoelastic/ folder is missing. Contact the project maintainer."
)

for model_name in ["elastic", "thermoelastic"]:
    model_dir = os.path.join(_HERE, "models", model_name)
    check(
        f"models/{model_name}/model_config.json",
        os.path.isfile(os.path.join(model_dir, "model_config.json")),
        f"Missing file: models/{model_name}/model_config.json"
    )
    check(
        f"models/{model_name}/normalization_stats.npz",
        os.path.isfile(os.path.join(model_dir, "normalization_stats.npz")),
        f"Missing file: models/{model_name}/normalization_stats.npz"
    )
    ckpt_dir = os.path.join(model_dir, "ckpt")
    has_ckpt = os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 0
    check(
        f"models/{model_name}/ckpt/ has checkpoint",
        has_ckpt,
        f"Checkpoint folder empty or missing: models/{model_name}/ckpt/"
    )

# ── 7. Load surrogate models ──────────────────────────────────────────────────
print("\n=== 7. Load surrogate models (end-to-end test) ===")

try:
    sys.path.insert(0, _HERE)
    sys.path.insert(0, el_dir)

    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    from forward import load_forward
    import jax.numpy as jnp

    for model_name in ["elastic", "thermoelastic"]:
        try:
            model = load_forward(model_name)
            n_in  = len(model.input_fields)
            n_out = len(model.output_fields)
            dummy = jnp.zeros(n_in, dtype=jnp.float32)
            out   = model.predict_array(dummy).block_until_ready()
            check(
                f"Load + predict '{model_name}' ({n_in} inputs → {n_out} outputs)",
                out.shape == (n_out,)
            )
        except Exception as e:
            check(f"Load + predict '{model_name}'", False, str(e))

except Exception as e:
    check("forward.py import", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if errors_found:
    print("  Some checks FAILED. Fix the issues above, then re-run this script.")
else:
    print("  All checks PASSED!  You are ready to run the GUIs.")
    print()
    print("  Forward GUI : python gui.py")
    print("  Inverse GUI : python gui_inverse.py")
print("=" * 60 + "\n")
