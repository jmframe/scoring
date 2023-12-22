from scipy.stats import skewnorm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import brier_score_loss
import numpy as np

# Parameters for noise
std_dev = 0.5  # Standard deviation of noise
skewness = 0.3  # Skewness of noise (0 for symmetric distribution)

# Generate observations
n = 1000
obs_bounds = [0, 1]
observations = np.array([obs_bounds[0] + np.random.random() * obs_bounds[1] for i in range(n)])
# Generate forecasts with normally distributed noise
forecasts = np.array([obs + skewnorm.rvs(a=skewness, loc=0, scale=std_dev) for obs in observations])
# Generating true binary outcomes (0 or 1)
true_binary_outcomes = np.random.randint(0, 2, size=n)
# Generating predicted probabilities with noise
predicted_probabilities = np.clip([skewnorm.rvs(a=skewness, loc=true_binary_outcomes[i], scale=std_dev) for i in range(n)], 0, 1)

# Calculate MSE
mse = mean_squared_error(observations, forecasts)
print("Mean Squared Error:", mse)

# Calculate Brier Score
brier_score = brier_score_loss(true_binary_outcomes, predicted_probabilities)

print("Brier Score:", brier_score)

def quantile_score(observations, forecasts, quantile):
    """
    Calculate the quantile score for a set of observations and quantile forecasts.

    :param observations: array-like, true observed values.
    :param forecasts: array-like, forecasted quantile values.
    :param quantile: float, the quantile for which the forecast is made (e.g., 0.5 for median).
    :return: float, the quantile score.
    """
    errors = observations - forecasts
    return np.mean((quantile - (errors < 0)) * errors)

quantile = 0.5  # Median quantile

score = quantile_score(observations, forecasts, quantile)

print("Quantile Score:", score)
