import jax.numpy as jnp
import numpy as np
import ml_collections

def get_dataset(config: ml_collections.ConfigDict):
    # Load dataset
    numpy_array = np.genfromtxt("two_variable_polynomial_dataset.csv", delimiter=',', skip_header=1)
    dataset = jnp.array(numpy_array)

    # Check expected shape
    expected_cols = config.input_dim + config.output_dim
    if dataset.shape[1] != expected_cols:
        raise ValueError(
            f"Dataset shape mismatch! Expected {expected_cols} columns (inputs + outputs), "
            f"but got {dataset.shape[1]} columns."
        )

    return dataset