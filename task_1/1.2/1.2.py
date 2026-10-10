
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("crime_train.csv")
test = pd.read_csv("crime_test.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)

answer = {"No": 0, "Yes": 1}
train["closed"] = train["closed"].map(answer)
test["closed"] = test["closed"].map(answer)

print("Share of closed cases in train:", round(train["closed"].mean(), 4))
print("Share of closed cases in test: ", round(test["closed"].mean(), 4))
print()

print("Missing values in train:")
print(train.isna().sum()[train.isna().sum() > 0])
print()

print("Odd values in train:")
print("  age above 100:", (train["age"] > 100).sum())
print("  police_department outside 1 to 19:",
      ((train["police_department"] < 1) | (train["police_department"] > 19)).sum())
print()

# share of closed cases for each value of a few columns
for col in ["sex", "weapon", "domain"]:
    print(train.groupby(col, dropna=False)["closed"].mean().round(3))
    print()

for df in [train, test]:
    df["month"] = df["case_filed"].str[0:2].astype(int)
    df["day"] = df["case_filed"].str[3:5].astype(int)
    df["year"] = df["case_filed"].str[6:10].astype(int)
    df["hour"] = df["case_filed"].str[11:13].astype(int)
    df["weapon"] = df["weapon"].fillna("None")
    df["age"] = df["age"].clip(upper=100)
    df["police_department"] = df["police_department"].clip(lower=1, upper=19)

train = train.drop(columns=["Unnamed: 0", "Num", "case_filed", "domain"])
test = test.drop(columns=["Unnamed: 0", "Num", "case_filed", "domain"])

number_columns = ["area", "age", "police_department", "month", "day", "year", "hour"]
word_columns = ["city", "crime_description", "sex", "weapon"]

print("Missing values left in train:", train.isna().sum().sum())
print("Missing values left in test: ", test.isna().sum().sum())
print("age range now:", train["age"].min(), "to", train["age"].max())
print("police_department range now:", train["police_department"].min(), "to", train["police_department"].max())

for col in word_columns:
    for value in train[col].unique():
        name = col + "_" + value
        train[name] = (train[col] == value).astype(int)
        test[name] = (test[col] == value).astype(int)
    train = train.drop(columns=[col])
    test = test.drop(columns=[col])
    train = train.copy()
    test = test.copy()

feature_columns = [c for c in train.columns if c != "closed"]
print("Number of input columns:", len(feature_columns))

mean = train[number_columns].mean()
std = train[number_columns].std()
train[number_columns] = (train[number_columns] - mean) / std
test[number_columns] = (test[number_columns] - mean) / std

X_train = train[feature_columns].to_numpy(dtype=float)
y_train = train["closed"].to_numpy(dtype=float)
X_test = test[feature_columns].to_numpy(dtype=float)
y_test = test["closed"].to_numpy(dtype=float)

# add the column of 1s for the bias
X_train = np.c_[np.ones(len(X_train)), X_train]
X_test = np.c_[np.ones(len(X_test)), X_test]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w):
    return sigmoid(X @ w)

def predict(X, w):
    return (predict_proba(X, w) >= 0.5).astype(int)

def accuracy(X, y, w):
    return (predict(X, w) == y).mean() * 100

learning_rate = 0.1
epochs = 300

w = np.zeros(X_train.shape[1])
m = len(y_train)

train_accs = []
test_accs = []

for epoch in range(1, epochs + 1):
    # one step of gradient descent (uses ONLY the train data)
    errors = predict_proba(X_train, w) - y_train
    gradient = (X_train.T @ errors) / m
    w = w - learning_rate * gradient

    # record the accuracy after this epoch
    train_accs.append(accuracy(X_train, y_train, w))
    test_accs.append(accuracy(X_test, y_test, w))

    print(f"Epoch {epoch:3d} | train accuracy: {train_accs[-1]:.2f}% | "
          f"test accuracy: {test_accs[-1]:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_accs, label="Train accuracy")
plt.plot(range(1, epochs + 1), test_accs, label="Test accuracy", linestyle="--")
plt.axhline(50, color="gray", linestyle=":", label="50% (just guessing)")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.ylim(40, 60)
plt.legend()
plt.savefig("accuracy_plot.png", dpi=150)
plt.show()

print("Final train accuracy:", round(train_accs[-1], 2), "%")
print("Final test accuracy: ", round(test_accs[-1], 2), "%")

# the baseline: always predict the most common class in the train data
baseline_class = 1 if y_train.mean() >= 0.5 else 0
print("Baseline test accuracy (always guess the most common class):",
      round((y_test == baseline_class).mean() * 100, 2), "%")


test_probability = predict_proba(X_test, w)
test_prediction = predict(X_test, w)

results = pd.DataFrame({
    "Expected closed": np.where(y_test == 1, "Yes", "No"),
    "Predicted closed": np.where(test_prediction == 1, "Yes", "No"),
    "Probability of closed": test_probability,
})
results["Correct?"] = np.where(y_test == test_prediction, "yes", "no")

print("Expected vs Predicted (first 20 rows of the test file):")
print(results.head(20).round(3).to_string())

results.to_csv("test_predictions.csv", index=False)
print("All", len(results), "test predictions saved to test_predictions.csv")

# how many of each kind of right and wrong answer
true_yes = ((test_prediction == 1) & (y_test == 1)).sum()
true_no = ((test_prediction == 0) & (y_test == 0)).sum()
false_yes = ((test_prediction == 1) & (y_test == 0)).sum()
false_no = ((test_prediction == 0) & (y_test == 1)).sum()

print()
print("========== FINAL SUMMARY ==========")
print("Epochs trained:", epochs)
print("Learning rate:", learning_rate)
print("Train accuracy:", round(accuracy(X_train, y_train, w), 2), "%")
print("Test accuracy: ", round(accuracy(X_test, y_test, w), 2), "%")
print("Test cases predicted correctly:", true_yes + true_no, "out of", len(y_test))
print("  predicted Yes and it was Yes:", true_yes)
print("  predicted No and it was No:  ", true_no)
print("  predicted Yes but it was No: ", false_yes)
print("  predicted No but it was Yes: ", false_no)