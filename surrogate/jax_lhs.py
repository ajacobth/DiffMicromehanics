import jax
import jax.numpy as jnp
from jax import grad, jit, lax, pmap
import argparse
import numpy as np
import pandas as pd
import time  # Import time module for timingdd

def lhs_jax(n_dim, criterion='m', iterations=5, keys=None, n_samples_per_device=1):
    """
    Parallelized Latin Hypercube Sampling (LHS) using `jax.pmap`.
    Each device generates a subset of samples to distribute memory load.
    """
    # Ensure `n_samples_per_device` is a Python integer (static)
    n_samples_per_device = int(n_samples_per_device)

    # Generate intervals (static size)
    intervals = jnp.linspace(0, 1, n_samples_per_device, endpoint=False)

    def sample_dim(k):
        perm = jax.random.permutation(k, intervals)
        if criterion == 'center':
            return perm + 0.5 / n_samples_per_device
        offsets = jax.random.uniform(k, (n_samples_per_device,)) / n_samples_per_device
        return perm + offsets

    # Vectorized sampling across dimensions (faster execution)
    samples = jax.vmap(sample_dim)(keys)

    if criterion in ['maximin', 'm']:
        print('Maximin criterion implemented')
        samples = _optimize_maximin(samples, iterations)

    return samples.T  # Transpose to (n_samples_per_device, n_dim)

@jit
def _optimize_maximin(samples, iterations):
    """Maximin optimization using JAX gradients with vectorized distance computation."""
    def loss(x):
        diff = x[:, None, :] - x[None, :, :]
        dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + jnp.eye(x.shape[0]) * 1e5)
        return -jnp.min(dists)

    def body_fn(i, val):
        samples = val
        grads = grad(loss)(samples)
        samples = jnp.clip(samples - 0.01 * grads, 0, 1)
        return samples

    samples = lax.fori_loop(0, iterations, body_fn, samples)
    return samples


@jit
def scale_lhs(samples, bounds):
    """Scale LHS samples to custom bounds using vectorized operations."""
    return samples * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def main():
    parser = argparse.ArgumentParser(description="Generate Parallelized Latin Hypercube Samples with JAX (pmap)")
    parser.add_argument("n_dim", type=int, help="Number of dimensions")
    parser.add_argument("n_samples", type=int, help="Total number of samples")
    parser.add_argument(
        "bounds",
        type=float,
        nargs="+",
        help="Bounds for each dimension as a flattened list [min1 max1 min2 max2 ...]",
    )

    args = parser.parse_args()

    if len(args.bounds) != 2 * args.n_dim:
        print(
            f"Error: Number of bounds provided ({len(args.bounds)}) "
            f"does not match expected ({2 * args.n_dim}). Please provide exactly two bounds per dimension."
        )
        return

    bounds_array = np.array(args.bounds).reshape(args.n_dim, 2)

    # Get available devices for parallel computation
    num_devices = jax.device_count()
    print(f"Using {num_devices} devices for parallel computation")

    # Ensure number of samples is divisible by devices
    if args.n_samples % num_devices != 0:
        print(f"Error: Number of samples ({args.n_samples}) must be divisible by {num_devices} devices.")
        return

    n_samples_per_device = args.n_samples // num_devices

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, num_devices * args.n_dim).reshape(num_devices, args.n_dim, 2)

    # Start timer
    start_time = time.time()

    # Fix: Pass `n_samples_per_device` as a static argument
    parallel_lhs = pmap(
        lambda n_dim, keys: lhs_jax(n_dim, 'm', 200, keys, n_samples_per_device),
        in_axes=(None, 0),
    )

    samples = parallel_lhs(args.n_dim, keys)

    # Reshape results back to (n_samples, n_dim)
    samples = samples.reshape(args.n_samples, args.n_dim)

    # Scale samples to bounds
    scaled_samples = scale_lhs(samples, bounds_array)

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Save to CSV
    df = pd.DataFrame(scaled_samples, columns=[f"Dim_{i+1}" for i in range(args.n_dim)])
    df.to_csv("lhs_samples_parallel.csv", index=False)

    print(f"LHS samples generated and saved to lhs_samples_parallel.csv")
    print(f"Execution Time: {elapsed_time:.4f} seconds")  # Display elapsed time


if __name__ == "__main__":
    main()
