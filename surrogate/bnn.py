import argparse
import os


os.environ["JAX_PLATFORM_NAME"] = "cpu"
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

from flax import linen as nn

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

# Import the flax_module helper from numpyro.contrib.module.
from numpyro.contrib.module import flax_module

# Import Flax for defining neural network modules.
from flax import linen as nn

matplotlib.use("Agg")  # Use non-interactive backend for plotting

# Define a Flax MLP model.
# This MLP implements a feedforward network with two hidden layers using tanh activations.
class MLP(nn.Module):
    layers: list[int]

    @nn.compact
    def __call__(self, x):
        for num_features in self.layers[:-1]:
            x = nn.tanh(nn.Dense(features=num_features)(x))
        return nn.Dense(features=self.layers[-1])(x)



def model(X, Y=None):
    mu_nn = flax_module("mu_nn", MLP(layers=[20, 20, 1]), input_shape=(1,))
    log_sigma_nn = flax_module("sigma_nn", MLP(layers=[2, 1]), input_shape=(1,))

    mu = numpyro.deterministic("mu", mu_nn(X).squeeze())
    sigma = numpyro.deterministic("sigma", nn.softplus(log_sigma_nn(X).squeeze()))


    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("likelihood", dist.Normal(loc=mu, scale=sigma), obs=Y)
        
        
# Bayesian neural network model using the Flax module.

# Generate artificial regression data.
def generate_data(n=100, key=random.PRNGKey(0)):
    # Uniform inputs in [-3, 3]
    X = jax.random.uniform(key, (n, 1), minval=-3.0, maxval=3.0)
    # A sine function with noise.
    Y = jnp.sin(X) + 0.1 * jax.random.normal(key, (n, 1))
    return X, Y.squeeze()

def main(args):
    rng_key = random.PRNGKey(0)
    X, Y = generate_data(n=args.num_data, key=rng_key)

    # Set up and run MCMC inference using NUTS.
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    print("Running MCMC...")
    mcmc.run(rng_key, X=X, Y=Y)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # Posterior predictive: generate predictions using the posterior samples.
    predictive = predictive = Predictive(model, posterior_samples=samples, num_samples=100)
    preds = predictive(rng_key, X=X)["likelihood"]
    mean_pred = jnp.mean(preds, axis=0)
    lower = jnp.percentile(preds, 5, axis=0)
    upper = jnp.percentile(preds, 95, axis=0)

    # Plot the training data and predictions.
    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, "kx", label="Data")
    plt.plot(X, mean_pred, "b-", label="Mean prediction")
    plt.fill_between(X.squeeze(), lower, upper, color="lightblue", alpha=0.5, label="90% CI")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bayesian Neural Network with Flax + NumPyro")
    plt.legend()
    plt.savefig("flax_numpyro_regression.pdf")
    print("Plot saved to flax_numpyro_regression.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flax + NumPyro Bayesian Neural Network Regression")
    parser.add_argument("--num-data", type=int, default=100, help="Number of training data points")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of MCMC samples")
    parser.add_argument("--num-warmup", type=int, default=10000, help="Number of warmup steps")
    parser.add_argument("--num-chains", type=int, default=1, help="Number of MCMC chains")
    parser.add_argument("--device", type=str, default="cpu", help='Use "cpu" or "gpu".')
    args = parser.parse_args()

    #numpyro.set_platform(args.device)
    #numpyro.set_host_device_count(args.num_chains)
    main(args)
