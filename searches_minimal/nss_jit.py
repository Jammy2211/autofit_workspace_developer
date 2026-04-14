"""
Minimal NSS Example (JAX JIT)
------------------------------

This script shows how to use the NSS (Nested Slice Sampling) JAX-based
nested sampler with a pure JAX likelihood, enabling full jax.jit
compilation for GPU acceleration.

The Gaussian model and likelihood are written entirely in jax.numpy so
the JIT-compiled inner loop runs without Python callbacks.

Requirements:
    pip install git+https://github.com/yallup/nss.git
    (pulls handley-lab/blackjax fork with nested sampling support)
"""
import time

import jax
import jax.numpy as jnp
from nss.ns import run_nested_sampling
from blackjax.ns.utils import log_weights

# --------------------------------------------------------------------------
# Data (generated with NumPy, converted to JAX arrays)
# --------------------------------------------------------------------------

import numpy as np

np.random.seed(1)

xvalues = jnp.arange(100, dtype=jnp.float64)
true_centre = 50.0
true_normalization = 25.0
true_sigma = 10.0

true_data = jnp.multiply(
    jnp.divide(true_normalization, true_sigma * jnp.sqrt(2.0 * jnp.pi)),
    jnp.exp(-0.5 * jnp.square(jnp.divide(xvalues - true_centre, true_sigma))),
)
noise_map = jnp.full(true_data.shape, 0.01)
data = true_data + jnp.array(np.random.normal(0.0, 0.01, true_data.shape))

# --------------------------------------------------------------------------
# Pure JAX model and likelihood
# --------------------------------------------------------------------------

# Parameter order: [centre, normalization, sigma]
# Priors: centre ~ U(0, 100), normalization ~ U(0, 50), sigma ~ U(0, 50)

prior_lower = jnp.array([0.0, 0.0, 0.0])
prior_upper = jnp.array([100.0, 50.0, 50.0])


def log_prior(params):
    """Uniform box prior in log-density form."""
    in_bounds = jnp.all((params >= prior_lower) & (params <= prior_upper))
    log_vol = jnp.sum(jnp.log(prior_upper - prior_lower))
    return jnp.where(in_bounds, -log_vol, -jnp.inf)


def log_likelihood(params):
    """Chi-squared log-likelihood for a 1D Gaussian model."""
    centre, normalization, sigma = params[0], params[1], params[2]
    model_data = jnp.multiply(
        jnp.divide(normalization, sigma * jnp.sqrt(2.0 * jnp.pi)),
        jnp.exp(-0.5 * jnp.square(jnp.divide(xvalues - centre, sigma))),
    )
    residuals = data - model_data
    chi_squared = jnp.sum(jnp.square(residuals / noise_map))
    return -0.5 * chi_squared


# --------------------------------------------------------------------------
# Run NSS
# --------------------------------------------------------------------------

n_live = 500
n_dim = 3

# Draw initial samples from the uniform prior.
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)
initial_samples = jax.random.uniform(
    init_key, shape=(n_live, n_dim), minval=prior_lower, maxval=prior_upper
)

print("Running NSS (JAX JIT) nested sampling...")
print(f"  n_live={n_live}, n_dim={n_dim}")
print(f"  JIT compilation will happen on first step (may take a moment)\n")

t_start = time.time()
final_state, results = run_nested_sampling(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=5,
    initial_samples=initial_samples,
    num_delete=1,
    termination=-3,
)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

# Extract weighted posterior samples.
logw = log_weights(rng_key, final_state)
positions = final_state.particles.position
log_likelihoods = final_state.particles.loglikelihood

# Best fit = highest likelihood sample.
best_idx = jnp.argmax(log_likelihoods)
best_params = positions[best_idx]

# Weighted mean from posterior.
w = jnp.exp(logw.mean(axis=-1))
w = w / w.sum()
weighted_mean = jnp.sum(positions * w[:, None], axis=0)

print(f"\n--- NSS (JAX JIT) Results ---")
print(f"Best fit:  centre={best_params[0]:.4f}  normalization={best_params[1]:.4f}  sigma={best_params[2]:.4f}")
print(f"True:      centre=50.0000  normalization=25.0000  sigma=10.0000")
print(f"Log evidence:  {results.logZs.mean():.2f}")

print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s")
print(f"  (includes JIT compilation)")
print(f"Sampling time:      {results.time:.2f} s")
print(f"  (excludes JIT warmup)")
print(f"Likelihood evals:   {int(results.evals)}")
print(f"Time per eval:      {results.time / int(results.evals) * 1e3:.3f} ms")
print(f"ESS:                {results.ess}")
print(f"Total samples:      {len(positions)}")
