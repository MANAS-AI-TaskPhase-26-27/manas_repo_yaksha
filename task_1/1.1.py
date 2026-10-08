import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------
# STEP 1: LOAD BOTH FILES
# ---------------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Train shape:", train.shape)
print("Test shape:", test.shape)


# ---------------------------------------------------
# STEP 2: CHOOSE THE COLUMNS
# 'close' is what we want to predict.
# We skip 'date' and 'index' because they don't help predict the price.
# ---------------------------------------------------
number_columns = [
    "high", "low", "momentum_index", "beta_indicator", "risk_premium",
    "volatility_factor", "technical_score", "oscillator_value",
    "liquidity_ratio", "open", "quant_index", "trend_strength",
    "market_sentiment", "volume", "alpha_signal",
]


# ---------------------------------------------------
# STEP 3: CLEAN THE DATA
# The file has empty cells AND fake 0 values where data is missing.
# So: turn the 0s into empty cells, then fill them with the median.
# The median comes from the TRAIN file only.
# ---------------------------------------------------
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


# ---------------------------------------------------
# STEP 4: TURN 'symbols' (WORDS) INTO 0/1 COLUMNS
# ---------------------------------------------------
symbol_columns = []
for value in train["symbols"].unique():
    name = "symbol_" + value
    train[name] = (train["symbols"] == value).astype(int)
    test[name] = (test["symbols"] == value).astype(int)
    symbol_columns.append(name)

feature_columns = number_columns + symbol_columns


# ---------------------------------------------------
# STEP 5: SPLIT INTO INPUTS (X) AND ANSWER (y)
# ---------------------------------------------------
X_train = train[feature_columns]
y_train = train["close"]

X_test = test[feature_columns]
y_test = test["close"]


# ---------------------------------------------------
# STEP 6: TRAIN THE MODEL (using the train file)
# ---------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)


# ---------------------------------------------------
# STEP 7: PREDICT THE CLOSE PRICE FOR THE TEST FILE
# ---------------------------------------------------
predictions = model.predict(X_test)


# ---------------------------------------------------
# STEP 8: CHECK HOW GOOD THE PREDICTIONS ARE
# ---------------------------------------------------
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Accuracy (R2 score):", round(r2 * 100, 2), "%")
print("Average error (MAE):", round(mae, 2))
print("RMSE:", round(rmse, 2))