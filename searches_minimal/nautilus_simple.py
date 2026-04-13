"""
Minimal Nautilus Example
------------------------

This script shows how to use the Nautilus nested sampler directly with
autofit's Model and Analysis objects, bypassing the full NonLinearSearch
wrapper class.

This is useful for quickly testing a search on a problem before investing
in a full autofit integration.

Requirements:
    pip install nautilus-sampler
"""
import numpy as np
import autofit as af

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
# Direct Nautilus interface
# --------------------------------------------------------------------------

from nautilus import Sampler


def prior_transform(cube):
    """Map unit cube to physical parameters via the model's priors."""
    return np.array(model.vector_from_unit_vector(cube))


def log_likelihood(params):
    """Evaluate log likelihood for a physical parameter vector."""
    instance = model.instance_from_vector(vector=params)
    return analysis.log_likelihood_function(instance)


sampler = Sampler(
    prior=prior_transform,
    likelihood=log_likelihood,
    n_dim=model.prior_count,
    n_live=200,
)

sampler.run(verbose=True)

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

points, log_w, log_l = sampler.posterior()

best_idx = np.argmax(log_l)
best_params = points[best_idx]
best_instance = model.instance_from_vector(vector=best_params)

print("\n--- Nautilus Results ---")
print(f"Centre:        {best_instance.centre:.2f}  (true: 50.0)")
print(f"Normalization: {best_instance.normalization:.2f}  (true: 25.0)")
print(f"Sigma:         {best_instance.sigma:.2f}  (true: 10.0)")
print(f"Log evidence:  {sampler.log_z:.2f}")
