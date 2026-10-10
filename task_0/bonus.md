Data Preprocessing of ManipalRains.csv

Step 1

```python
import pandas as pd

df = pd.read_csv("ManipalRains.csv")
print("Start shape:", df.shape)
```

Load the file into a table called df. It has 145,460 rows and 23 columns.


Step 2

```python
df = df.dropna(subset=["RainTomorrow"])
print("After removing empty RainTomorrow:", df.shape)
```

Explanation: dropna deletes rows that have an empty cell. Because I wrote subset=["RainTomorrow"], it only looks at that one column.

Problem: RainTomorrow is the answer the model has to learn to predict. Some rows have no answer. A model cannot learn from a question that has no answer.

Solution: Delete those rows. 3,267 rows were removed and 142,193 are left.


Step 3

```python
df["Day"] = df["Date"].str[0:2].astype(int)
df["Month"] = df["Date"].str[3:5].astype(int)
df["Year"] = df["Date"].str[6:10].astype(int)
df = df.drop(columns=["Date"])
```

Explanation: The date looks like 01-12-2008. The slicing [0:2] takes the first two characters ("01"), [3:5] takes the month ("12") and [6:10] takes the year ("2008"). astype(int) changes the text into a real number. The last line deletes the old Date column.

Problem: The date is text, and a model cannot understand text like 01-12-2008.

Solution: Split it into three number columns: Day, Month and Year. The month is useful because rain depends on the season.


Step 4

```python
number_columns = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]

for col in number_columns:
    middle = df[col].median()
    df[col] = df[col].fillna(middle)
```

Explanation: First I made a list of all the number columns. Then a for loop goes through them one by one. For each column it finds the median (the middle value) and uses fillna to put that value in every empty cell.

Problem: Many number columns have empty cells, and a model cannot work with empty cells. Deleting all those rows would throw away too much data.

Solution: Fill each empty cell with the median of its column. I used the median and not the mean because extreme values (like one huge rainfall day) pull the mean but barely change the median.


Step 5

```python
text_columns = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]

for col in text_columns:
    most_common = df[col].mode()[0]
    df[col] = df[col].fillna(most_common)
```

Explanation: Same idea as step 4, but for text columns. mode() finds the value that appears the most, and [0] picks it. fillna puts that value in the empty cells.

Problem: Text columns also have empty cells, and you cannot find the middle of words like NW or SE.

Solution: Fill the empty cells with the most common value of that column.


Step 6

```python
yes_no = {"No": 0, "Yes": 1}
df["RainToday"] = df["RainToday"].map(yes_no)
df["RainTomorrow"] = df["RainTomorrow"].map(yes_no)
```

Explanation: yes_no is a dictionary that says No means 0 and Yes means 1. map goes through the column and replaces every word using that dictionary.

Problem: RainToday and RainTomorrow contain the words Yes and No. A model only understands numbers.

Solution: Change No to 0 and Yes to 1.


Step 7

```python
word_columns = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]

for col in word_columns:
    for value in df[col].unique():
        new_name = col + "_" + value
        df[new_name] = (df[col] == value).astype(int)
    df = df.drop(columns=[col])
    df = df.copy()
```

Explanation: The outer loop goes through each word column. The inner loop goes through every different value in that column (unique gives them). For each value it makes a new column, for example Location_Albury. The part (df[col] == value) is True when the row has that value, and astype(int) turns True into 1 and False into 0. After that, the old word column is deleted. The copy line just tidies the table and stops a pandas warning.

Problem: Location and the wind direction columns still contain words. If I numbered them (Albury = 0, Bega = 1), the model would think the cities have an order and that one is bigger than another, which is not true.

Solution: Make one 0/1 column for each value. This is called one-hot encoding. This step adds 97 columns (49 locations and 16 for each of the three wind columns), which is why the data ends up with 118 columns.


Step 8

```python
for col in number_columns:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std
```

Explanation: For each number column I find the mean (average) and the std (standard deviation, which shows how spread out the values are). Then every value is changed using the formula (value - mean) / std.

Problem: The columns are on very different scales. Pressure is around 1000 and rainfall is around 2. The big numbers can take over the model just because they are big.

Solution: Scale every number column so it has a mean of 0 and a standard deviation of 1. This is called standardization. Now all columns are on a similar scale.


Step 9

```python
print("Final shape:", df.shape)
print("Empty cells left:", df.isna().sum().sum())
print("Text columns left:", list(df.select_dtypes(exclude="number").columns))
```

Explanation: The first line prints the size of the table. The second counts all the empty cells that are left. The third lists any columns that are not numbers.

Problem: I need to be sure the cleaning really worked.

Solution: Check the output. I got 142,193 rows and 118 columns, 0 empty cells and an empty list of text columns. So the data is clean and fully numeric.


Step 10

```python
df.to_csv("ManipalRains_clean.csv", index=False)
print("Saved!")
```

Explanation: to_csvcd  saves the table as a new CSV file. index=False stops pandas from adding an extra column of row numbers.

Problem: The cleaned data only exists in Python's memory and is lost when the program ends.

Solution: Save it as ManipalRains_clean.csv. The original file is not changed.