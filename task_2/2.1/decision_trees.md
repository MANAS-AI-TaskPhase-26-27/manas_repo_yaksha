# Decision Trees and Ensemble Methods

## 1. Decision trees

A decision tree is a nonlinear model that divides the feature space into regions using a sequence of questions about individual features. Each internal node applies a split, each branch represents an outcome of that split, and each leaf gives a prediction.

Tree construction uses **top-down, greedy, recursive partitioning**:

- **Top-down:** begin with the full input space.
- **Recursive:** split a region, then treat each child region as a smaller problem.
- **Greedy:** at each step, choose the split that gives the best immediate improvement. The procedure does not search through every possible complete tree.

A numerical split is usually defined by a feature (j) and threshold (t). Within a parent region (R_p), it creates child regions such as:

\[
R_1 = \{x \in R_p : x_j < t\}, \qquad
R_2 = \{x \in R_p : x_j \ge t\}.
\]

The purpose of a split is to create child regions in which predictions are more reliable than a prediction made over the whole parent region.

## 2. Classification leaves and split criteria

For classification, a leaf predicts the most common class among the training observations that reach it. Let \(\hat p_c\) be the proportion of observations in a region that belong to class (c).

### Misclassification loss

The misclassification loss of a region is the fraction that would be classified incorrectly if the region predicted its majority class:

\[
L_{\text{misclass}}(R) = 1 - \max_c \hat p_c.
\]

A candidate split is assessed by comparing the parent loss with the **observation-weighted** losses of its children:

\[
\text{gain} = L(R_p) - \left(w_1 L(R_1) + w_2 L(R_2)\right),
\]

where (w_i) is the proportion of the parent observations that go to child (i). A useful split has a large positive gain.

Misclassification loss is not very sensitive to changes in class proportions as long as the majority class stays the same. Two splits can therefore have the same misclassification loss even if one creates more informative class distributions. This makes misclassification loss less useful for choosing splits during tree growth.

### Entropy / cross-entropy

Entropy is a more sensitive measure of class mixture:

\[
L_{\text{entropy}}(R) = -\sum_c \hat p_c \log_2(\hat p_c).
\]

Entropy is zero when a region contains only one class. It is larger when the class proportions are more evenly mixed. The information-theoretic interpretation is that a mixed region requires more information to identify an observation’s class.

The lecture uses base 2 to express information in bits. Using another logarithm base only multiplies entropy by a positive constant, so it does not change which split is best.

Because the parent class proportions are the weighted average of the children’s proportions, the parent entropy can be compared with the weighted average child entropy. The reduction is the **information gain** of the split.

### Gini impurity

Gini impurity is another common split criterion:

\[
L_{\text{Gini}}(R) = \sum_c \hat p_c(1-\hat p_c)
= 1 - \sum_c \hat p_c^2.
\]

Like entropy, Gini is low for nearly pure regions and higher for mixed regions. Both criteria are generally more informative for split selection than misclassification loss.

## 3. Regression trees and categorical features

### Regression trees

A regression tree predicts a numerical target. Each leaf predicts the mean target value among its training observations:

\[
\hat y_m = \frac{1}{|R_m|}\sum_{i\in R_m} y_i.
\]

Splits are chosen to reduce within-region prediction error, commonly the sum of squared deviations from the leaf mean:

\[
\sum_{i\in R_m}(y_i - \hat y_m)^2.
\]

### Categorical features

A categorical feature can be split by testing whether its value belongs to a selected subset of categories. With (q) categories, exhaustively considering subsets can require examining exponentially many possibilities, on the order of (2^q). This becomes expensive for features with many categories. Some special cases allow more efficient searches; for binary classification, categories can be ordered by their positive-class proportions and candidate splits searched along that ordering.

## 4. Tree complexity and regularization

An unrestricted tree can continue splitting until it memorizes the training data. This can produce low training error but poor generalization. Trees are therefore often **high variance**: their fitted structure can change substantially when the training sample changes.

Common controls on tree growth include:

- **Minimum leaf size:** stop splitting a region when it has too few observations.
- **Maximum depth:** limit the number of splits along any path.
- **Maximum number of nodes or leaves:** limit the total size of the tree.
- **Pruning:** grow a tree and then remove branches that do not improve performance on held-out data.

A strict minimum-improvement rule can stop a tree too early. Some patterns require multiple splits: an initial split may provide little immediate benefit, while a later split within one child may be valuable. Pruning after growing the tree can preserve the opportunity to find such multi-step structure. The lecture describes using validation performance to guide pruning.

## 5. Computation and properties of a single tree

Let (n) be the number of observations, (f) the number of features, and (d) the tree depth.

- Prediction follows one path from root to leaf and takes roughly (O(d)) operations.
- A rough training-time estimate from the lecture is (O(nfd)), assuming efficient searches over feature split candidates.
- A balanced binary tree often has depth on the order of \(\log n\), while an unbalanced tree can be deeper.

**Strengths:** trees can model nonlinear decision boundaries and feature interactions, can handle categorical features, are relatively fast, and can be interpreted by following their sequence of splits.

**Limitations:** trees can overfit and have high variance. Their splits are axis-aligned, so they can need many regions to approximate boundaries that do not align with individual feature axes. A single tree may consequently have limited predictive accuracy.

## 6. Why ensembles help

An ensemble combines predictions from multiple models. Averaging can reduce variance when the models do not make exactly the same errors.

For (M) models with equal variance \(\sigma^2\) and pairwise correlation \(\rho\), the variance of their average is:

\[
\operatorname{Var}(\bar X) = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2.
\]

This simplified expression highlights two ways to reduce ensemble variance:

1. Increase the number (M) of models, reducing the second term.
2. Reduce correlation \(\rho\) among the models, reducing the first term.

If models are independent, averaging reduces variance to \(\sigma^2/M\). If they are perfectly correlated, averaging does not reduce variance. Model diversity is therefore important.

Ensembles can create diversity by using different learning algorithms, different training samples, or procedures such as bagging and boosting that vary how component models are fit.

## 7. Bagging

**Bagging** stands for **bootstrap aggregation**. It approximates training on multiple datasets by repeatedly resampling the available training set.

For each of (M) bootstrap samples:

1. Draw (n) observations from the original training set of size (n).
2. Sample **with replacement**, so some observations can appear multiple times and some may be left out.
3. Fit a separate model to that sample.

Each bootstrap sample contains about 63% unique observations from the original dataset on average. The samples overlap, so the resulting models are related rather than independent.

For regression, bagging averages the component predictions:

\[
\hat f_{\text{bag}}(x) = \frac{1}{M}\sum_{m=1}^{M}\hat f_m(x).
\]

For classification, predictions can be combined by majority vote or by averaging class probabilities.

Bagging primarily reduces variance by averaging models trained on slightly different samples. It can slightly increase bias because each component model is fit to a resampled dataset, but the variance reduction is often especially useful for high-variance models such as trees. Increasing the number of bootstrap models can improve stability, though gains level off because the samples are drawn from the same finite dataset and remain correlated.

## 8. Random forests

A random forest combines bagging with additional randomization in the tree-building process. At each split, a tree considers only a randomly selected subset of the features, then chooses the best split among that subset.

This reduces the chance that every tree repeatedly chooses the same dominant feature at the same point in its structure. The resulting trees are less correlated, which can make averaging them more effective.

- **Bagging:** randomizes the training observations used for each tree.
- **Random forests:** randomize the training observations and the candidate features considered at each split.

## 9. Boosting

Boosting builds an ensemble sequentially. Each new model is fit after the earlier models and is directed toward what the current ensemble handles poorly. Component predictions are then added together, usually with weights.

The lecture presents boosting as an additive approach that can reduce bias: several simple learners can combine into a more expressive predictor. This contrasts with bagging, where models are fit independently and then averaged, mainly reducing variance.

### Decision stumps and AdaBoost

A **decision stump** is a tree with one split. In the AdaBoost-style process described in the lecture:

1. Fit a weak learner to the weighted training data.
2. Increase emphasis on observations it misclassifies.
3. Fit the next learner using the updated weights.
4. Give each learner a contribution weight based on its performance.
5. Add the weighted learners to form the final model.

A general additive form is:

\[
G(x) = \sum_{m=1}^{M} \alpha_m G_m(x),
\]

where (G_m) is the learner from round (m) and \(\alpha_m\) is its contribution weight. In binary classification, the final class is typically determined from the sign of the weighted sum.

For standard AdaBoost, the learner weight is commonly written as

\[
\alpha_m = \frac{1}{2}\ln\left(\frac{1-\epsilon_m}{\epsilon_m}\right),
\]

where \(\epsilon_m\) is the learner’s weighted classification error. The central concept is that training weights change over rounds and the learners are combined additively.

Gradient boosting and XGBoost are related boosting methods. They also build additive models in stages, but their precise objectives and update rules differ from AdaBoost.

## 10. Bagging, random forests, and boosting

| Method | How component models differ | How predictions are combined | Main effect emphasized in the lecture |
|---|---|---|---|
| Bagging | Bootstrap samples of the training observations | Average or vote | Reduces variance |
| Random forest | Bootstrap samples and random feature subsets at each split | Average or vote | Reduces variance and model correlation |
| Boosting | Sequential fitting, with later learners focused on current errors | Weighted additive sum | Reduces bias by building a stronger model from weak learners |

## Core ideas to retain

- Trees build predictions through greedy, recursive splits of the feature space.
- Classification splits are chosen by reducing weighted child impurity; entropy and Gini are more sensitive than misclassification loss.
- Regression leaves predict means, with splits commonly judged by squared error.
- Limiting tree depth, leaf size, or overall size helps prevent overfitting; pruning uses held-out performance to remove unhelpful branches.
- A single tree is interpretable and flexible, but often unstable and high variance.
- Averaging helps when component models are both numerous and not too correlated.
- Bagging uses bootstrap samples to reduce variance; random forests further decorrelate trees by restricting candidate features at each split.
- Boosting fits models sequentially and combines their weighted predictions, with later learners addressing errors left by earlier ones.
