"""
Minimal NSS Example (autofit interface)
----------------------------------------

This script shows how to use the NSS (Nested Slice Sampling) JAX-based
nested sampler directly with autofit's Model and Analysis objects,
bypassing the full NonLinearSearch wrapper class.

The NumPy-based likelihood from autofit is wrapped with jax.pure_callback
so that it can be called inside NSS's JIT-compiled sampling loop.

Requirements:
    pip install git+https://github.com/yallup/nss.git
    (pulls handley-lab/blackjax fork with nested sampling support)
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import autofit as af
from nss.ns import run_nested_sampling
from blackjax.ns.utils import log_weights

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class Gaussian:
    def __init__(self, centre=30.0, normalization=1.0, sigma=5.0):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_from(self, xvalues):
        return np.multiply(
            np.divide(self.normalization, self.sigma * np.sqrt(2.0 * np.pi)),
            np.exp(-0.5 * np.square(np.divide(xvalues - self.centre, self.sigma))),
        )


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

xvalues = np.arange(100)
true_gaussian = Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
data = true_gaussian.model_data_from(xvalues=xvalues)
noise_map = np.full(data.shape, 0.01)
data += np.random.normal(0.0, 0.01, data.shape)

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()
        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)
        return log_likelihood


analysis = Analysis(data=data, noise_map=noise_map)

# --------------------------------------------------------------------------
# autofit Model (provides priors and parameter mapping)
# --------------------------------------------------------------------------

model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

# --------------------------------------------------------------------------
# Direct NSS interface
# --------------------------------------------------------------------------

n_likelihood_calls = 0


def numpy_log_likelihood(params_np):
    """Evaluate log likelihood via autofit (NumPy)."""
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=params_np.tolist())
    return np.float64(analysis.log_likelihood_function(instance))


def numpy_log_prior(params_np):
    """Evaluate log prior via autofit (NumPy)."""
    log_priors = model.log_prior_list_from_vector(vector=params_np.tolist())
    return np.float64(sum(log_priors))


def log_likelihood(params):
    """JAX wrapper around the NumPy likelihood via pure_callback."""
    return jax.pure_callback(
        lambda p: jnp.float64(numpy_log_likelihood(np.asarray(p))),
        jax.ShapeDtypeStruct((), jnp.float64),
        params,
        vmap_method="sequential",
    )


def log_prior(params):
    """JAX wrapper around the NumPy prior via pure_callback."""
    return jax.pure_callback(
        lambda p: jnp.float64(numpy_log_prior(np.asarray(p))),
        jax.ShapeDtypeStruct((), jnp.float64),
        params,
        vmap_method="sequential",
    )


# Draw initial samples from the uniform prior.
# n_live is kept small because each likelihood call goes through a Python
# callback (jax.pure_callback), which is much slower than pure JAX.
# See nss_jit.py for the fast, pure-JAX version.
n_live = 50
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)

prior_lower = jnp.array([0.0, 0.0, 0.0])
prior_upper = jnp.array([100.0, 50.0, 50.0])
initial_samples = jax.random.uniform(
    init_key, shape=(n_live, model.prior_count), minval=prior_lower, maxval=prior_upper
)

print("Running NSS (autofit interface) nested sampling...")
print(f"  n_live={n_live}, n_dim={model.prior_count}")
print(f"  Using jax.pure_callback for NumPy likelihood\n")

t_start = time.time()
final_state, results = run_nested_sampling(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=2,
    initial_samples=initial_samples,
    num_delete=10,
    termination=-1,
)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

positions = final_state.particles.position
log_likelihoods = final_state.particles.loglikelihood

best_idx = jnp.argmax(log_likelihoods)
best_params = positions[best_idx]
best_instance = model.instance_from_vector(vector=np.asarray(best_params).tolist())

print(f"\n--- NSS (autofit) Results ---")
print(f"Best fit:  centre={best_instance.centre:.4f}  normalization={best_instance.normalization:.4f}  sigma={best_instance.sigma:.4f}")
print(f"True:      centre=50.0000  normalization=25.0000  sigma=10.0000")
print(f"Log evidence:  {results.logZs.mean():.2f}")

print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s")
print(f"  (includes JIT compilation)")
print(f"Sampling time:      {results.time:.2f} s")
print(f"  (excludes JIT warmup)")
print(f"Likelihood calls:   {n_likelihood_calls}")
print(f"Time per call:      {t_elapsed / n_likelihood_calls * 1e3:.3f} ms")
print(f"ESS:                {results.ess}")
print(f"Total samples:      {len(positions)}")
