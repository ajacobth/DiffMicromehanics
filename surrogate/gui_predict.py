import os
import ml_collections
import tkinter as tk
from tkinter import messagebox
import jax.numpy as jnp
import numpy as np
import models
from NN_surrogate.utils import restore_checkpoint

# Function to forcefully close previous Tkinter instances
def force_close_tk():
    try:
        root = tk.Tk()
        root.destroy()  # Close any existing Tkinter windows
    except tk.TclError:
        pass  # Ignore errors if no instance exists

# GUI Class
class ModelGUI:
    def __init__(self, root, config, model, params, input_mean, input_std, target_mean, target_std):
        self.root = root
        self.root.title("JAX MF Prediction")

        self.num_inputs = config.input_dim
        self.num_outputs = config.output_dim
        self.model = model
        self.params = params
        self.input_mean = input_mean
        self.input_std = input_std
        self.target_mean = target_mean
        self.target_std = target_std

        self.input_entries = []

        # Title Label
        tk.Label(root, text="JAX Model Prediction", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

        # Create input fields
        tk.Label(root, text="Enter Inputs:", font=("Helvetica", 12, "bold")).grid(row=1, column=0, columnspan=2, pady=5)
        for i in range(self.num_inputs):
            tk.Label(root, text=f"Input {i+1}:", font=("Helvetica", 10)).grid(row=i+2, column=0, sticky="e", padx=5)
            entry = tk.Entry(root)
            entry.grid(row=i+2, column=1, padx=5, pady=2)
            self.input_entries.append(entry)

        # Predict Button
        self.predict_button = tk.Button(root, text="Predict", font=("Helvetica", 12, "bold"), command=self.predict)
        self.predict_button.grid(row=self.num_inputs + 2, column=0, columnspan=2, pady=10)

        # Output Section
        tk.Label(root, text="Predicted Outputs:", font=("Helvetica", 12, "bold")).grid(row=self.num_inputs + 3, column=0, columnspan=2, pady=5)
        self.output_labels = []
        for i in range(self.num_outputs):
            lbl = tk.Label(root, text=f"Output {i+1}: --", font=("Helvetica", 10), fg="white")
            lbl.grid(row=self.num_inputs + 4 + i, column=0, columnspan=2, pady=2)
            self.output_labels.append(lbl)

    def predict(self):
        try:
            raw_inputs = [float(entry.get()) for entry in self.input_entries]
            raw_inputs = jnp.array(raw_inputs)

            norm_inputs = (raw_inputs - self.input_mean) / self.input_std
            norm_outputs = self.model.u_net(self.params, norm_inputs.reshape(1, -1))

            outputs = (norm_outputs * self.target_std) + self.target_mean

            for i, lbl in enumerate(self.output_labels):
                lbl.config(text=f"Output {i+1}: {outputs.flatten()[i]:.4f}")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

# Main GUI Function
def gui(config: ml_collections.ConfigDict, workdir: str):
    # Ensure previous Tkinter instances are closed
    force_close_tk()

    model = models.MICRO_SURROGATE(config)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    state = restore_checkpoint(model.state, ckpt_path)
    params = state['params']

    norm_stats_path = os.path.join(workdir, "normalization_stats.npz")
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f"Normalization statistics file not found: {norm_stats_path}")

    norm_stats = np.load(norm_stats_path)

    input_mean = jnp.array(norm_stats["input_mean"])
    input_std = jnp.array(norm_stats["input_std"])
    target_mean = jnp.array(norm_stats["target_mean"])
    target_std = jnp.array(norm_stats["target_std"])

    # Ensure GUI runs properly
    root = tk.Tk()
    app = ModelGUI(root, config, model, params, input_mean, input_std, target_mean, target_std)
    root.mainloop()
