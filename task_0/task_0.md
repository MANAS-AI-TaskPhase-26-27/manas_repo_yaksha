# INTRODUCTION TO MACHINE LEARNING

## 1. Overview of AI/ML

Artificial Intelligence (AI) is the broad field of building systems that perform tasks normally requiring human intelligence. Machine Learning (ML) is a subset of AI: instead of hand-coding rules, we train a piece of software (a **model**) on data so it learns patterns and can make predictions or generate content on its own.

Example: to predict rainfall the "traditional" way, you'd need to hand-build a physics simulation of the atmosphere. The ML way is to feed a model years of weather data and let it learn the relationship between weather patterns and rainfall itself, then apply that learned relationship to new data.

ML systems generally fall into four categories:

- **Supervised learning** :learns from labeled data (data with known correct answers)
- **Unsupervised learning** :finds patterns/structure in unlabeled data
- **Reinforcement learning** :learns by trial and error, getting rewards/penalties for actions in an environment
- **Generative AI** :learns the underlying patterns in data well enough to generate new, original content (text, images, audio, video, code)

## 2. Supervised vs. Unsupervised Learning

**Supervised learning**: the model is trained on examples that already have the correct answer attached (like studying old exam papers with answer keys). It learns the mapping from inputs to outputs.
- **Regression** :predicts a continuous number (e.g., predicted house price, rainfall in mm)
- **Classification** :predicts a category/class (e.g., spam vs. not spam). Can be *binary* (2 classes) or *multiclass* (3+ classes)

**Unsupervised learning**: the model is given data with no labels and has to find structure on its own :most commonly through **clustering**, grouping similar data points together. Unlike classification, the categories aren't defined in advance by a human; you often have to interpret and label the clusters after the fact (e.g., clustering weather data might naturally separate into groups that correspond to "rain," "snow," "hail," "no rain" :but the model didn't know those labels going in).

**Key distinction**: supervised = "here's the answer key, learn from it." Unsupervised = "find the pattern yourself, no answer key given."

## 3. Training, Validation, and Test Sets

A dataset is normally split into three parts:

| Split | Purpose |
|---|---|
| **Training set** | The data the model actually learns from :it repeatedly sees these examples and adjusts itself to reduce error |
| **Validation set** | Used *during* development to tune the model (choose settings/hyperparameters, decide when to stop training) without touching the test set |
| **Test set** | Held back until the very end, used once to get an honest, unbiased measure of how the model performs on data it has never seen |

The reason for splitting: if you evaluate a model on the same data it trained on, it will look artificially good :it may have just memorized the answers rather than learned a generalizable pattern. Keeping validation/test data separate is what lets you catch that.

## 4. What Is a "Model" and How Does Training Work?

A **model** is essentially a mathematical relationship (a large collection of numbers/parameters) that maps input features to an output :a label or prediction. It doesn't "know" anything in advance; it starts with a rough/random relationship and improves it through training.

**Training**, at a high level:
1. The model takes a labeled example and makes a prediction
2. It compares its prediction to the actual/true value :the difference is called the **loss**
3. Based on that loss, the model adjusts its internal parameters slightly to reduce future error
4. This repeats across the entire dataset, often many times over (multiple "epochs")
5. Over many iterations, the model's predictions get closer to the true values, meaning it has learned the underlying relationship between features and label

After training, the model is **evaluated** on labeled data it hasn't adjusted itself on, to see how well it generalizes. Once performance is acceptable, the model is used for **inference** :making predictions on brand-new, unlabeled, real-world data.

Two dataset qualities matter a lot for training quality:
- **Size** :number of examples
- **Diversity** :how wide a range of situations the examples cover

A large dataset isn't automatically a good one :e.g., 100 years of data but only from July won't help predict January rainfall. You need both scale and variety.

## 5. Why Data Needs to Be Cleaned and Preprocessed

Raw, real-world data is almost never ready to feed directly into a model. It typically has missing values, inconsistent formats, outliers, non-numeric categories, and features on wildly different scales. If left as-is:
- The model may learn incorrect or misleading patterns
- Some algorithms will simply error out on missing/non-numeric data
- Features with larger numeric ranges can unfairly dominate the learning process

Preprocessing turns messy raw data into a clean, consistent, numeric format the model can actually learn from effectively :"garbage in, garbage out" applies directly to ML.

## 6. Handling Missing Data (Imputation)

Common strategies:
- **Deletion** :drop rows (or columns) with missing values, only sensible when missingness is rare and random
- **Mean/median/mode imputation** :fill numeric gaps with the column's mean or median (median is more robust to outliers); fill categorical gaps with the most frequent category
- **Constant/placeholder value** :fill with a fixed value like 0 or "Unknown," sometimes combined with a separate "was this missing?" indicator column
- **Forward/backward fill** :for time-series data, carry the last known (or next known) value forward/backward
- **Model-based imputation** :predict the missing value using other features (e.g., regression or k-nearest-neighbors imputation)

The right strategy depends on *why* data is missing (random vs. systematic) and how much of it is missing.

## 7. Handling Outliers

Outliers are data points far outside the normal range that can distort training (especially for models sensitive to scale, like linear regression).

Common approaches:
- **Detect** them first :e.g., via box plots, z-scores, or the IQR (interquartile range) method
- **Remove** them if they're clearly data-entry errors or not representative of the real-world process
- **Cap/clip (winsorize)** them to a reasonable min/max boundary instead of deleting
- **Transform** the data (e.g., log transform) to reduce the influence of extreme values
- **Use robust models/metrics** that are naturally less sensitive to outliers (e.g., median-based methods, tree-based models)

Not all outliers are "bad data" :sometimes they're the most important signal (e.g., fraud detection), so context matters before removing them.

## 8. Categorical Encoding

Most ML models need numeric input, so non-numeric categories (like "red/blue/green" or "city name") must be converted to numbers:

- **One-hot encoding** :creates a separate binary (0/1) column for each category; best for categories with no inherent order (nominal data)
- **Label/ordinal encoding** :assigns each category an integer (0, 1, 2...); appropriate when categories have a natural order (e.g., "low/medium/high")
- **Target/mean encoding** :replaces a category with a statistic (like the average label value) for that category; powerful but risks data leakage if not done carefully
- **Frequency encoding** :replaces a category with how often it appears in the data

Choosing the wrong method (e.g., label-encoding an unordered category) can accidentally introduce a false sense of order that misleads the model.

## 9. Feature Scaling

Features often come in very different ranges (e.g., "age" in tens, "income" in tens of thousands). Many algorithms (especially distance-based ones like k-NN, or gradient-based ones) perform poorly or train slowly if features aren't on comparable scales.

Common techniques:
- **Normalization (min-max scaling)** :rescales values into a fixed range, typically [0, 1]
- **Standardization (z-score scaling)** :rescales values to have a mean of 0 and standard deviation of 1
- **Robust scaling** :uses median and IQR instead of mean/std, so it's less affected by outliers

Tree-based models (like decision trees/random forests) are generally scale-invariant and don't strictly need this step, but most other models benefit from or require it.

## 10. Overfitting vs. Underfitting

- **Overfitting**: the model learns the training data *too* well :including its noise and quirks :so it performs great on training data but poorly on new, unseen data. It has memorized rather than generalized.
- **Underfitting**: the model is too simple to capture the underlying pattern at all, so it performs poorly on both training and new data.

The goal is a model that generalizes well :good performance on unseen data. Common fixes:
- **For overfitting**: get more/more diverse data, simplify the model, use regularization, use cross-validation, use early stopping
- **For underfitting**: use a more complex/expressive model, add more relevant features, train longer

## 11. Evaluation Metrics

**For classification:**
- **Accuracy** :% of predictions that were correct overall. Misleading on imbalanced datasets (e.g., 99% accuracy is meaningless if 99% of data is one class)
- **Precision** :of everything the model predicted as positive, what fraction was actually positive? (Matters when false positives are costly)
- **Recall (sensitivity)** :of everything that was actually positive, what fraction did the model correctly catch? (Matters when false negatives are costly)
- **F1 score** :harmonic mean of precision and recall; balances the two into a single number
- **Confusion matrix** :a table breaking down true positives, true negatives, false positives, and false negatives
- **ROC-AUC** :measures how well the model separates classes across different decision thresholds

**For regression:**
- **MAE (Mean Absolute Error)** :average absolute difference between predicted and actual values
- **MSE / RMSE (Mean/Root Mean Squared Error)** :squares the errors before averaging (penalizes large errors more); RMSE brings it back to the original units
- **R² (R-squared)** :proportion of variance in the target explained by the model (closer to 1 is better)

The right metric depends on the problem :e.g., in medical diagnosis, recall (catching all real positive cases) often matters more than raw accuracy.

## 12. Key Takeaways

- ML = learning patterns from data instead of hand-coding rules
- Supervised = has labels (answers); Unsupervised = no labels, find structure
- Split data into train/validation/test to get an honest measure of performance
- A model is just a learned mathematical relationship; training gradually reduces its prediction error (loss)
- Clean, well-preprocessed data (handled missing values, outliers, encoded categories, scaled features) is essential :bad input data breaks or biases the model
- Watch for the overfitting/underfitting balance :the goal is generalization, not memorization
- Pick evaluation metrics based on what actually matters for the problem, not just accuracy by default
