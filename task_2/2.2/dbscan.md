# DBSCAN Clustering

## What DBSCAN does

DBSCAN is a density-based clustering algorithm. It forms clusters from connected regions of high point density and treats sufficiently isolated points as noise. Unlike K-Means, it does not require the number of clusters in advance and can find clusters with non-spherical or nested shapes.

The name stands for **Density-Based Spatial Clustering of Applications with Noise**.

## Two parameters

- **`eps` (epsilon):** the radius used to define a point's neighborhood. Two points are neighbors if their distance is at most `eps`.
- **`min_samples`:** the minimum number of points needed in a neighborhood for a point to count as a **core point**.

Parameter conventions can differ across implementations: `min_samples` commonly counts the point itself as part of its neighborhood. Check the convention used by the implementation.

These settings control what counts as dense. A smaller `eps` or larger `min_samples` makes it harder for points to be core points; this can create more noise or split clusters. A larger `eps` or smaller `min_samples` can join more points and may merge clusters.

## Point types

- **Core point:** has at least `min_samples` points in its `eps` neighborhood.
- **Border point:** is not a core point, but lies within the neighborhood of a core point.
- **Noise point / outlier:** is not a core point and is not within the neighborhood of any core point.

A border point can join a cluster, but it does not grow or extend that cluster. Core points do extend clusters by connecting to neighboring core points.

## How DBSCAN builds clusters

1. Find the `eps` neighborhood of each point.
2. Identify core points by checking whether each neighborhood meets `min_samples`.
3. Choose an unvisited core point and start a cluster.
4. Add its neighboring core points to the cluster.
5. Keep expanding through connected neighboring core points. This finds the full density-connected region.
6. Add border points that are close to core points in that cluster. Do not use border points to expand the cluster.
7. Start another cluster from any remaining unassigned core point.
8. Mark points that are not part of any cluster as noise.

The algorithm can be viewed as finding connected components among core points, then attaching nearby border points.

## Border points near multiple clusters

Clusters are discovered in sequence. A border point can be within `eps` of core points from more than one cluster. In that case, it may be assigned to whichever cluster reaches it first. The core structure of the clusters is unchanged, but this can make the final border assignment depend on traversal order.

## Distance and multiple features

DBSCAN relies on distances between observations. Euclidean distance in (p) dimensions is:

\[
d(x,y)=\sqrt{\sum_{j=1}^{p}(x_j-y_j)^2}.
\]

The method works in higher-dimensional feature spaces without plotting the data. However, distance-based methods are sensitive to feature scales, so features are often standardized before calculating distances.

## DBSCAN compared with K-Means

- K-Means requires (K) and assigns every observation to a cluster.
- DBSCAN does not require (K), can identify outliers, and can find clusters with irregular shapes.
- DBSCAN depends on the density parameters `eps` and `min_samples`; poor choices can merge groups, split them, or classify many points as noise.

## Stuff

- DBSCAN defines clusters through dense, connected neighborhoods.
- Core points create and extend clusters; border points can join but cannot extend them.
- Noise points are not reachable from any core point.
- `eps` and `min_samples` determine what the algorithm considers dense.
- A border point close to multiple clusters may be assigned based on cluster discovery order.
- Scale features appropriately before using distance-based clustering.
