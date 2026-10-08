import pandas as pd

# ---------------------------------------------------
# STEP 1: LOAD THE FILE
# ---------------------------------------------------
df = pd.read_csv("ManipalRains.csv")
print("Start shape:", df.shape)


# ---------------------------------------------------
# STEP 2: REMOVE ROWS WHERE RainTomorrow IS EMPTY
# (we need the answer to train, so these rows are useless)
# ---------------------------------------------------
df = df.dropna(subset=["RainTomorrow"])
print("After removing empty RainTomorrow:", df.shape)


# ---------------------------------------------------
# STEP 3: FIX THE DATE
# Date looks like "01-12-2008"  ->  day-month-year
# Cut it into pieces using slicing, turn each into int
# ---------------------------------------------------
df["Day"] = df["Date"].str[0:2].astype(int)
df["Month"] = df["Date"].str[3:5].astype(int)
df["Year"] = df["Date"].str[6:10].astype(int)
df = df.drop(columns=["Date"])


# ---------------------------------------------------
# STEP 4: FILL EMPTY NUMBER CELLS WITH THE MEDIAN
# ---------------------------------------------------
number_columns = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]

for col in number_columns:
    middle = df[col].median()
    df[col] = df[col].fillna(middle)


# ---------------------------------------------------
# STEP 5: FILL EMPTY TEXT CELLS WITH THE MOST COMMON VALUE
# ---------------------------------------------------
text_columns = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]

for col in text_columns:
    most_common = df[col].mode()[0]
    df[col] = df[col].fillna(most_common)


# ---------------------------------------------------
# STEP 6: TURN Yes/No INTO 1/0
# ---------------------------------------------------
yes_no = {"No": 0, "Yes": 1}
df["RainToday"] = df["RainToday"].map(yes_no)
df["RainTomorrow"] = df["RainTomorrow"].map(yes_no)


# ---------------------------------------------------
# STEP 7: TURN OTHER WORDS INTO NUMBERS (one-hot, by hand)
# For every value (like "NW") make a new column:
# 1 if the row has that value, 0 if not.
# ---------------------------------------------------
word_columns = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]

for col in word_columns:
    for value in df[col].unique():
        new_name = col + "_" + value
        df[new_name] = (df[col] == value).astype(int)
    df = df.drop(columns=[col])
    df = df.copy()  # tidies the table (stops a pandas warning)


# ---------------------------------------------------
# STEP 8: SCALE THE NUMBER COLUMNS BY HAND
# new value = (value - mean) / std
# ---------------------------------------------------
for col in number_columns:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std


# ---------------------------------------------------
# STEP 9: CHECK EVERYTHING WORKED
# ---------------------------------------------------
print("Final shape:", df.shape)
print("Empty cells left:", df.isna().sum().sum())
print("Text columns left:", list(df.select_dtypes(exclude="number").columns))


# ---------------------------------------------------
# STEP 10: SAVE
# ---------------------------------------------------
df.to_csv("ManipalRains_clean.csv", index=False)
print("Saved!")