import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

number_columns = [
    "high", "low", "momentum_index", "beta_indicator", "risk_premium",
    "volatility_factor", "technical_score", "oscillator_value",
    "liquidity_ratio", "open", "quant_index", "trend_strength",
    "market_sentiment", "volume", "alpha_signal",
]

for col in number_columns:
    train[col] = train[col].replace(0, np.nan)
    test[col] = test[col].replace(0, np.nan)

medians = train[number_columns].median()

train[number_columns] = train[number_columns].fillna(medians)
test[number_columns] = test[number_columns].fillna(medians)

# We can't learn from (or check against) a row with no real close price.
train = train[train["close"].notna() & (train["close"] != 0)]
test = test[test["close"].notna() & (test["close"] != 0)]
print("Train rows after cleaning:", train.shape[0])
print("Test rows after cleaning:", test.shape[0])

symbol_columns = []
for value in train["symbols"].unique():
    name = "symbol_" + value
    train[name] = (train["symbols"] == value).astype(int)
    test[name] = (test["symbols"] == value).astype(int)
    symbol_columns.append(name)

feature_columns = number_columns + symbol_columns

X_train = train[feature_columns].to_numpy(dtype=float)
y_train = train["close"].to_numpy(dtype=float)

X_test = test[feature_columns].to_numpy(dtype=float)
y_test = test["close"].to_numpy(dtype=float)


mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

# don't scale y (close).

# prediction = X @ w            (weighted sum of the inputs)
# cost       = (1 / m) * sum((prediction - y)^2)   (mean squared error)
# accuracy   = R2 score

def predict(X, w):
    return X @ w


def cost(X, y, w):
    m = len(y)
    errors = predict(X, w) - y
    return (errors ** 2).sum() / m


def r2_score(y_real, y_pred):
    ss_res = ((y_real - y_pred) ** 2).sum()
    ss_tot = ((y_real - y_real.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

# TRAIN WITH GRADIENT DESCENT
#   1. predict with the current weights
#   2. find the errors (prediction - real)
#   3. find the gradient = (2/m) * X.T @ errors
#   4. move the weights a small step against the gradient
# After every epoch we save the cost and the accuracy.
# The test accuracy is only for watching progress.

learning_rate = 0.1
epochs = 300

w = np.zeros(X_train.shape[1])
m = len(y_train)

train_costs = []
test_costs = []
train_accs = []
test_accs = []

for epoch in range(1, epochs + 1):
    # one step of gradient descent
    errors = predict(X_train, w) - y_train
    gradient = 2 * (X_train.T @ errors) / m
    w = w - learning_rate * gradient

    # save the cost
    train_costs.append(cost(X_train, y_train, w))
    test_costs.append(cost(X_test, y_test, w))

    # save the accuracy (predictions are already in real prices)
    train_pred = predict(X_train, w)
    test_pred = predict(X_test, w)
    train_accs.append(r2_score(y_train, train_pred) * 100)
    test_accs.append(r2_score(y_test, test_pred) * 100)

    print(f"Epoch {epoch:3d} | cost: {train_costs[-1]:.5f} | "
          f"train accuracy: {train_accs[-1]:.2f}% | "
          f"test accuracy: {test_accs[-1]:.2f}%")


plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_costs, label="Train cost")
plt.plot(range(1, epochs + 1), test_costs, label="Test cost", linestyle="--")
plt.title("Cost vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.yscale("log")
plt.legend()
plt.savefig("cost_plot.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_accs, label="Train accuracy")
plt.plot(range(1, epochs + 1), test_accs, label="Test accuracy", linestyle="--")
plt.title("Accuracy (R2 score) vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.legend()
plt.savefig("accuracy_plot.png", dpi=150)
plt.show()

final_pred = predict(X_test, w)
mae = np.abs(y_test - final_pred).mean()
rmse = np.sqrt(((y_test - final_pred) ** 2).mean())

print("Final train accuracy (R2):", round(train_accs[-1], 2), "%")
print("Final test accuracy  (R2):", round(test_accs[-1], 2), "%")
print("Average error (MAE):", round(mae, 2))
print("RMSE:", round(rmse, 2))

results = pd.DataFrame({
    "Expected close": y_test,
    "Predicted close": final_pred,
})
results["Error"] = results["Predicted close"] - results["Expected close"]

print("Expected vs Predicted (first 20 rows of the test file):")
print(results.head(20).round(2).to_string())

results.to_csv("test_predictions.csv", index=False)
print("All", len(results), "test predictions saved to test_predictions.csv")

train_final_pred = predict(X_train, w)
print()
print("========== FINAL SUMMARY ==========")
print("Epochs trained:", epochs)
print("Learning rate:", learning_rate)
print("Train cost (MSE):", round(cost(X_train, y_train, w), 2))
print("Test cost (MSE): ", round(cost(X_test, y_test, w), 2))
print("Train accuracy (R2):", round(r2_score(y_train, train_final_pred) * 100, 2), "%")
print("Test accuracy (R2): ", round(r2_score(y_test, final_pred) * 100, 2), "%")
print("Test MAE: ", round(mae, 2))
print("Test RMSE:", round(rmse, 2))