"""
Minimal NSS Example (JAX Grad + HMC)
--------------------------------------

This script shows how to use NSS's HMC-based Sequential Monte Carlo
variant, which leverages jax.grad for gradient-accelerated sampling.

The Gaussian model and likelihood are written in pure jax.numpy so that
automatic differentiation works. The HMC kernel uses gradients of the
log-posterior to propose moves, which can improve sampling efficiency
for smooth, well-behaved likelihoods.

Requirements:
    pip install git+https://github.com/yallup/nss.git
    (pulls handley-lab/blackjax fork with nested sampling support)
"""
import time

import jax
import jax.numpy as jnp
from nss.smc import run_hmc_sequential_mc, sample_smc

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
# Pure JAX model and likelihood (differentiable)
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


# Verify gradient works.
test_params = jnp.array([50.0, 25.0, 10.0])
grad_fn = jax.grad(log_likelihood)
test_grad = grad_fn(test_params)
print(f"Gradient check at true values: {test_grad}")
print(f"  (non-zero due to noise in synthetic data)\n")

# --------------------------------------------------------------------------
# Run NSS with HMC (gradient-based SMC)
# --------------------------------------------------------------------------

n_live = 500
n_dim = 3

# Draw initial samples from the uniform prior.
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)
initial_samples = jax.random.uniform(
    init_key, shape=(n_live, n_dim), minval=prior_lower, maxval=prior_upper
)

print("Running NSS (HMC + jax.grad) sequential Monte Carlo...")
print(f"  n_live={n_live}, n_dim={n_dim}")
print(f"  Using jax.grad for HMC gradient computation")
print(f"  JIT compilation will happen on first step (may take a moment)\n")

t_start = time.time()
smc_state, results = run_hmc_sequential_mc(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=5,
    initial_samples=initial_samples,
    hmc_trajectory_length=5,
    target_ess=0.9,
    warmup_steps=200,
)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

# Resample from the weighted SMC particles.
rng_key, sample_key = jax.random.split(rng_key)
posterior_samples = sample_smc(sample_key, smc_state, n=5000)

# Best fit = highest-weight particle.
best_idx = jnp.argmax(smc_state.weights)
best_params = jax.tree_util.tree_map(lambda x: x[best_idx], smc_state.particles)

# Also evaluate log-likelihood at best to find the actual ML estimate.
all_logl = jax.vmap(log_likelihood)(posterior_samples)
ml_idx = jnp.argmax(all_logl)
ml_params = posterior_samples[ml_idx]

# Posterior summary from resampled particles.
medians = jnp.median(posterior_samples, axis=0)
stds = jnp.std(posterior_samples, axis=0)
labels = ["centre", "normalization", "sigma"]

print(f"\n--- NSS (HMC + jax.grad) Results ---")
print(f"Best fit:  centre={ml_params[0]:.4f}  normalization={ml_params[1]:.4f}  sigma={ml_params[2]:.4f}")
print(f"True:      centre=50.0000  normalization=25.0000  sigma=10.0000")
print(f"Log evidence:  {results.logZs.mean():.2f}")

print(f"\nPosterior summary (median +/- 1 sigma):")
for i, label in enumerate(labels):
    print(f"  {label:15s} = {medians[i]:.4f} +/- {stds[i]:.4f}")

print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s")
print(f"  (includes JIT compilation + HMC warmup)")
print(f"Sampling time:      {results.time:.2f} s")
print(f"  (excludes JIT warmup)")
print(f"Gradient evals:     {int(results.evals)}")
print(f"  (each = likelihood + gradient via jax.grad)")
print(f"Time per eval:      {results.time / max(int(results.evals), 1) * 1e3:.3f} ms")
print(f"ESS:                {results.ess:.0f}")
print(f"Posterior samples:  {len(posterior_samples)}")
