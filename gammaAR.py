# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 21:38:34 2024

@author: kevin
"""


# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Function to generate an autoregressive gamma process
def generate_autoregressive_gamma_process(n, alpha, beta, phi):
    # Initialize arrays
    observations = np.zeros(n)
    
    # Generate observations
    for i in range(1, n):
        shape_parameter = alpha + phi * observations[i - 1]
        observations[i] = np.random.gamma(shape_parameter, scale=1/beta)
    
    return observations

# Parameters
n_samples = 1000
alpha_value = 2.0  # Shape parameter of the gamma distribution
beta_value = 1.0   # Scale parameter of the gamma distribution
phi_value = 0.5    # Autoregressive parameter

# Generate autoregressive gamma process
gamma_process = generate_autoregressive_gamma_process(n_samples, alpha_value, beta_value, phi_value)

# Plot the generated process
plt.plot(gamma_process, label='Autoregressive Gamma Process')
plt.title('Autoregressive Gamma Process')
plt.xlabel('Time')
plt.ylabel('Observations')
plt.legend()
plt.show()

# %%
from scipy.optimize import minimize

# Simulated data
# np.random.seed(42)
observed_data = generate_autoregressive_gamma_process(n_samples, alpha_value, beta_value, phi_value)

# Likelihood function for gamma-distributed data
def likelihood(params, data):
    alpha, beta, phi = params
    # n = len(data)
    shape_parameter = alpha + phi * data[:-1]
    log_likelihood = np.sum((shape_parameter-1)*np.log(data[1:]) - sp.special.gammaln(shape_parameter) - shape_parameter*np.log(beta) - data[1:]/beta)
    # log_likelihood = -n * np.log(beta) - np.sum(np.log(np.random.gamma(alpha + phi * data[:-1], scale=1/beta)))
    # log_likelihood = -n * np.log(beta) + np.sum((alpha + phi * data[:-1]) * np.log(data[1:]/beta) - data[1:]/beta - np.log(sp.special.gamma(alpha + phi * data[:-1])))
    # log_likelihood = 0
    # for i in range(1, n):
    #     shape_parameter = alpha + phi * data[i - 1]
    #     log_likelihood += (shape_parameter - 1) * np.log(data[i]/beta) - data[i]/beta - np.log(np.math.gamma(shape_parameter))

    return -log_likelihood  # Negative log-likelihood for minimization

# Initial parameter values
initial_params = [1.0, 2.0, 1.5]

# Maximize the likelihood using scipy.optimize.minimize
result = minimize(likelihood, initial_params, args=(observed_data,), method='L-BFGS-B')

# Extract the inferred parameters
inferred_alpha, inferred_beta, inferred_phi = result.x

print("Inferred Parameters:")
print("Alpha:", inferred_alpha)
print("Beta:", inferred_beta)
print("Phi:", inferred_phi)
