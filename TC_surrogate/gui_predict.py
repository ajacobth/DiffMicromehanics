import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import ml_collections
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------------
# Matplotlib: use Tk backend and turn off LaTeX so labels never break
# ----------------------------------------------------------------------------
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["text.usetex"] = False

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import models
from NN_surrogate.utils import restore_checkpoint

# ----------------------------------------------------------------------------
# Feature names – order matters
# ----------------------------------------------------------------------------

INPUT_FIELD_NAMES = [
    "k_f1", "k_f2", "k_m", "ar_f",
    "w_f", "rho_f", "rho_m",
    "a11", "a22", "a12", "a13", "a23", # a33 = 1 - a11 - a22
]

# The size of the inputs is automatically foudn during the code execution

#k_f1	k_f2	k_m	ar_f	w_f	rho_f	rho_m	a11	a22	a12	a13	a23

# k11	k12	k13	k22	k23	k33
OUTPUT_FIELD_NAMES = [
    "k11", "k12", "k13", "k22", "k23",
    "k33"]


# ----------------------------------------------------------------------------
def force_close_tk():
    """Close any orphan Tk windows (helps in IDE)."""
    try:
        tmp = tk.Tk(); tmp.destroy()
    except tk.TclError:
        pass


# ----------------------------------------------------------------------------
class ModelGUI:
    """Tk GUI with threaded inference + live plots."""

    def __init__(self, root, config, model, params,
                 input_mean, input_std, target_mean, target_std):

        # ==== store ====
        self.root        = root
        self.model       = model
        self.params      = params
        self.input_mean  = input_mean
        self.input_std   = input_std
        self.target_mean = target_mean
        self.target_std  = target_std

        # ==== history for plots ====
        self.step_idx = []
        self.y_hist   = [[] for _ in OUTPUT_FIELD_NAMES]

        # ---- main window ----
        root.title("Composite-MF Surrogate")
        root.geometry("1200x860")
        root.minsize(900, 700)

        # ------------------------------------------------------------------
        # 1. Input fields (left column)
        # ------------------------------------------------------------------
        tk.Label(root, text="Enter Inputs:",
                 font=("Helvetica", 20, "bold")
                 ).grid(row=0, column=0, sticky="w", padx=6, pady=(10, 4))

        self.input_entries = []
        for i, name in enumerate(INPUT_FIELD_NAMES):
            tk.Label(root, text=f"{name}:", font=("Helvetica", 20)
                     ).grid(row=i+1, column=0, sticky="e", padx=6)
            ent = tk.Entry(root, width=14)
            ent.grid(row=i+1, column=1, sticky="w", padx=3, pady=1)
            self.input_entries.append(ent)

        # Predict button & status label
        self.predict_btn = tk.Button(root, text="Predict",
                                     font=("Helvetica", 16, "bold"),
                                     command=self._predict_start)
        self.predict_btn.grid(row=len(INPUT_FIELD_NAMES)+1, column=0,
                              columnspan=2, pady=8)

        self.status_lbl = tk.Label(root, text="", fg="orange",
                                   font=("Helvetica", 16, "italic"))
        self.status_lbl.grid(row=len(INPUT_FIELD_NAMES)+2, column=0,
                             columnspan=2, sticky="w")

        # ------------------------------------------------------------------
        # 2. Latest numeric outputs (right column)
        # ------------------------------------------------------------------
        tk.Label(root, text="Elasticity Tensor Componets:",
                 font=("Helvetica", 20, "bold")
                 ).grid(row=0, column=2, columnspan=2, sticky="w", padx=6)

        self.output_labels = []
        for i, name in enumerate(OUTPUT_FIELD_NAMES):
            lbl = tk.Label(root, text=f"{name}: --", font=("Helvetica", 20))
            lbl.grid(row=1+i, column=2, columnspan=2,
                     sticky="w", padx=6, pady=1)
            self.output_labels.append(lbl)

        # ------------------------------------------------------------------
        # 3. Scrollable 3×3 plot grid
        # ------------------------------------------------------------------
        plot_frame = ttk.Frame(root)
        plot_frame.grid(row=len(INPUT_FIELD_NAMES)+3, column=0,
                        columnspan=4, sticky="nsew", padx=5, pady=5)
        root.grid_rowconfigure(len(INPUT_FIELD_NAMES)+3, weight=1)
        root.grid_columnconfigure(3, weight=1)

        canvas = tk.Canvas(plot_frame, borderwidth=0, highlightthickness=0)
        vbar   = ttk.Scrollbar(plot_frame, orient="vertical",
                               command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self.fig, axes = plt.subplots(3, 3,
                                      figsize=(10, 8),
                                      constrained_layout=True)
        self.axes  = axes.flatten()
        self.lines = []
        for ax, name in zip(self.axes, OUTPUT_FIELD_NAMES):
            line, = ax.plot([], [], marker="o")
            self.lines.append(line)
            ax.set_title(name)
            ax.set_xlabel("Prediction index")
            ax.set_ylabel("Value")

        self.canvas = FigureCanvasTkAgg(self.fig, master=inner)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # -----------------------------------------------------------------
    # Threaded prediction interface
    # -----------------------------------------------------------------
    def _predict_start(self):
        """Disable button, launch worker thread."""
        self.predict_btn.config(state="disabled")
        self.status_lbl.config(text="Running…")
        threading.Thread(target=self._predict_worker, daemon=True).start()

    def _predict_worker(self):
        """Runs in background thread, does the heavy JAX work."""
        try:
            raw = [float(e.get()) for e in self.input_entries]
            x   = jnp.array(raw)

            x_n = (x - self.input_mean) / self.input_std
            y_n = self.model.u_net(self.params, x_n.reshape(1, -1))
            y   = (y_n * self.target_std + self.target_mean
                   ).flatten().block_until_ready()
            y   = np.asarray(y, dtype=float)  # to host

            # schedule UI update on main thread
            self.root.after(0, lambda: self._update_ui(y))
        except Exception as exc:
            self.root.after(
                0,
                lambda: messagebox.showerror(
                    "Error", f"Prediction failed:\n {exc}")
            )
            self.root.after(0, self._reset_button)

    # ---------------- helpers ---------------------------------------
    def _update_ui(self, y_vals):
        # numeric labels
        for lbl, name, val in zip(self.output_labels,
                                  OUTPUT_FIELD_NAMES, y_vals):
            lbl.config(text=f"{name}: {val:.4f}")

        # history + plots
        step = len(self.step_idx) + 1
        self.step_idx.append(step)
        for hist, val in zip(self.y_hist, y_vals):
            hist.append(val)

        for line, ys, ax in zip(self.lines, self.y_hist, self.axes):
            line.set_data(self.step_idx, ys)
            ax.relim(); ax.autoscale_view()

        self.canvas.draw_idle()
        self._reset_button()

    def _reset_button(self):
        self.status_lbl.config(text="")
        self.predict_btn.config(state="normal")


# ----------------------------------------------------------------------------
def gui(config: ml_collections.ConfigDict, workdir: str):
    """Restore model & stats, warm-up compile, then launch GUI."""
    force_close_tk()

    # --- restore model ----------------------------------------------------
    model    = models.MICRO_SURROGATE(config)
    ckpt_dir = os.path.join(workdir, "ckpt", config.wandb.name)
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint dir: {ckpt_dir}")
    state  = restore_checkpoint(model.state, ckpt_dir)
    params = state["params"]

    # --- normalisation stats ---------------------------------------------
    stats_file = os.path.join(workdir, "normalization_stats.npz")
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"No stats file: {stats_file}")
    stats = np.load(stats_file)
    input_mean  = jnp.array(stats["input_mean"])
    input_std   = jnp.array(stats["input_std"])
    target_mean = jnp.array(stats["target_mean"])
    target_std  = jnp.array(stats["target_std"])

    # --- warm-up compile (eliminate first-click lag) ----------------------
    dummy = jnp.zeros((1, config.input_dim))
    _ = model.u_net(params, dummy).block_until_ready()
    print("JAX graph compiled; launching GUI…")

    # --- launch Tk interface ---------------------------------------------
    root = tk.Tk()
    _ = ModelGUI(root, config, model, params,
                 input_mean, input_std, target_mean, target_std)
    root.mainloop()
