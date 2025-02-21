
import jax.numpy as jnp
import numpy as np
import ml_collections

def get_dataset(config: ml_collections.ConfigDict):
    # Load 2D array using standard numpy
    numpy_array = np.load(config.data.path)
    # Convert to JAX numpy array with float32 precision
    dataset = jnp.array(numpy_array, dtype=jnp.float32)
    return dataset
