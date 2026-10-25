# K-Means Clustering

## What K-Means does

K-Means is an unsupervised learning method that groups observations into a chosen number of clusters, (K). It assigns each observation to the cluster with the nearest center, then updates each center to the mean of the observations assigned to it.

The process repeats until assignments stop changing (or another stopping condition is reached).

## The algorithm

1. Choose the number of clusters, (K).
2. Pick (K) distinct data points as the initial cluster centers.
3. Measure each observation’s distance to every center.
4. Assign each observation to its nearest center.
5. Recalculate each center as the mean of the observations in that cluster.
6. Repeat the assignment and update steps until the clusters settle.
7. Keep track of the total within-cluster variation and repeat the process with different starting centers. Return the result with the lowest variation found.

For a point (x_i) assigned to cluster (C_k), its center is:

\[
\mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k}x_i
\]

A common measure of clustering quality is within-cluster sum of squares (also called inertia):

\[
\text{Inertia} = \sum_{k=1}^{K}\sum_{x_i \in C_k}\|x_i-\mu_k\|^2
\]

Lower inertia means observations are closer to their assigned centers. But inertia always decreases as (K) increases, so it cannot be used by itself to choose (K).

## Why repeat with different starting centers?

The initial centers are chosen randomly, and a poor starting choice can lead K-Means to a poor final grouping. K-Means may settle on a local solution rather than the best possible clustering. Running it several times with different initial centers helps; the algorithm keeps the run with the lowest within-cluster variation.

## Choosing the number of clusters: the elbow method

Try several values of (K), calculate the within-cluster variation for each, and plot the results:

- Increasing (K) makes clusters smaller, so variation goes down.
- At (K=1), all observations are in one cluster.
- If every observation has its own cluster, variation reaches zero, but that is not a useful summary.
- Look for the point where adding more clusters stops reducing variation by much. This bend is the **elbow** and can be used as a reasonable choice for (K).

The elbow is a visual heuristic; sometimes the plot does not have a clear bend.

## K-Means with multiple features

K-Means is not limited to data that can be drawn on a line. Each observation can have two, three, or many features. The algorithm compares observations using a distance measure, commonly Euclidean distance:

\[
d(x,y)=\sqrt{\sum_{j=1}^{p}(x_j-y_j)^2}
\]

For two features this is the usual Pythagorean distance; adding more features adds more squared differences under the square root. The data do not need to be plotted to be clustered—the distances are what matter.

Because Euclidean distance is affected by feature scales, features with very different units or ranges are often standardized before clustering.

## K-Means and hierarchical clustering

- **K-Means** requires a chosen (K) and directly partitions observations into that many groups.
- **Hierarchical clustering** builds relationships between observations or groups based on similarity. It provides a hierarchy of merges rather than requiring the final number of clusters at the start.

## Stuff

- K-Means alternates between assigning points to their nearest center and recalculating centers as means.
- It stops when cluster assignments or centers stop changing appreciably.
- Different random initializations can produce different results, so multiple runs are useful.
- Within-cluster variation measures how tightly grouped the clusters are, but it decreases as (K) grows.
- The elbow plot helps balance tighter clusters against using too many clusters.
- K-Means works in any number of dimensions as long as distances can be calculated.
