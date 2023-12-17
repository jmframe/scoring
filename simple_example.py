from sklearn.metrics import mean_squared_error

# True values
y_true = [3, -0.5, 2, 7]

# Predicted values
y_pred = [2.5, 0.0, 2, 8]

# Calculate MSE
mse = mean_squared_error(y_true, y_pred)
print("y_true", y_true)
print("y_pred", y_pred)
print("Mean Squared Error:", mse)

from sklearn.metrics import brier_score_loss

# True binary outcomes (0 or 1)
true_binary_outcomes = [1, 0, 1, 1]

# Predicted probabilities of the positive class (1)
predicted_probabilities = [0.9, 0.4, 0.7, 0.8]

# Calculate Brier Score
brier_score = brier_score_loss(true_binary_outcomes, predicted_probabilities)

print("true_binary_outcomes", true_binary_outcomes)
print("predicted_probabilities", predicted_probabilities)
print("Brier Score:", brier_score)

import numpy as np

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

# Example usage
observations = np.array([2.3, 3.5, 4.6, 2.8])
forecasts = np.array([2.5, 3.7, 4.2, 3.0])
quantile = 0.5  # Median

score = quantile_score(observations, forecasts, quantile)
print("observations", observations)
print("forecasts", forecasts)
print("Quantile Score:", score)
