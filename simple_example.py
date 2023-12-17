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

