# K-Means Clustering: Centroid Initialization Methods Investigation

## Overview

This project investigates and compares three different centroid initialization methods used in the K-Means clustering algorithm:

1. **Random Data Points** - Simple random selection
2. **Naive Sharding** - Data partitioning approach
3. **K-Means++** - Intelligent probabilistic selection

The implementation includes a complete K-Means algorithm from scratch, comparison metrics, and visualizations to demonstrate how different initialization methods affect clustering performance.

---

## Table of Contents

1. [Introduction to K-Means Clustering](#introduction-to-k-means-clustering)
2. [Initialization Methods](#initialization-methods)
3. [Comparison: K-Means++ vs Random Initialization](#comparison-k-means-vs-random-initialization)
4. [Implementation Details](#implementation-details)
5. [Usage](#usage)
6. [Results and Analysis](#results-and-analysis)
7. [Conclusion](#conclusion)

---

## Introduction to K-Means Clustering

K-Means is an unsupervised machine learning algorithm used for clustering data into k groups. The algorithm works in two main steps:

1. **Assignment Step**: Assign each data point to the nearest centroid
2. **Update Step**: Recalculate centroids as the mean of all points in each cluster

The algorithm iterates between these steps until convergence (centroids no longer change significantly) or a maximum number of iterations is reached.

### The Initialization Problem

The quality of K-Means clustering heavily depends on the initial placement of centroids. Poor initialization can lead to:
- Suboptimal cluster assignments
- Slow convergence
- Inconsistent results across runs
- Convergence to local minima instead of global optimum

---

## Initialization Methods

### 1. Random Data Points Initialization

**How it works:**
- Randomly selects k data points from the dataset as initial centroids
- Each point has an equal probability of being chosen

**Algorithm:**
```
1. Randomly select k distinct indices from the dataset
2. Use the corresponding data points as initial centroids
```

**Advantages:**
- Simple and fast to implement
- Low computational overhead
- Easy to understand

**Disadvantages:**
- Can lead to poor clustering if random points are close together
- Inconsistent results across different runs
- May converge to local minima
- No guarantee of good initial spread

**Use Case:** Quick prototyping or when computational resources are limited

---

### 2. Naive Sharding Initialization

**How it works:**
- Divides the data into k equal-sized shards along the first dimension
- Calculates the mean (centroid) of each shard
- Uses these means as initial centroids

**Algorithm:**
```
1. Sort data points by their first feature
2. Divide sorted data into k equal shards
3. Calculate the mean of each shard
4. Use these means as initial centroids
```

**Advantages:**
- Ensures centroids are spread across the data space
- Deterministic (same data produces same initialization)
- Better than random for some datasets
- Fast computation

**Disadvantages:**
- Assumes clusters are aligned with the first dimension
- May not work well for non-linear or complex cluster shapes
- Ignores relationships between features
- Can fail if clusters are not separated along the first dimension

**Use Case:** When data has clear separation along one dimension

---

### 3. K-Means++ Initialization

**How it works:**
- Intelligently selects centroids to maximize spread
- Uses a probabilistic approach based on distance from existing centroids

**Algorithm:**
```
1. Choose the first centroid randomly from the dataset
2. For each remaining centroid (k-1 times):
   a. Calculate the squared distance from each point to its nearest existing centroid
   b. Select the next centroid with probability proportional to these squared distances
   c. Points farther from existing centroids have higher probability of being selected
```

**Advantages:**
- Produces better initial centroids than random selection
- Faster convergence (fewer iterations needed)
- More consistent and stable results
- Better cluster quality (lower within-cluster sum of squares)
- Proven theoretical guarantees (logarithmic approximation ratio)

**Disadvantages:**
- Slightly slower initialization than random (O(nkd) vs O(kd))
- More complex implementation
- Still not guaranteed to find global optimum

**Use Case:** Production systems, when cluster quality matters, standard practice in modern implementations

---

## Comparison: K-Means++ vs Random Initialization

### Performance Metrics

#### 1. **Cluster Quality (Inertia/WCSS)**

**Inertia (Within-Cluster Sum of Squares)** measures how tightly points are clustered around their centroids. Lower inertia indicates better clustering.

- **K-Means++**: Typically achieves 10-30% lower inertia than random initialization
- **Random**: Higher variance in results, often produces suboptimal clusters

**Why?** K-Means++ ensures centroids are well-spread, leading to more natural cluster boundaries.

#### 2. **Convergence Speed**

- **K-Means++**: Converges in 20-40% fewer iterations on average
- **Random**: May require many iterations, especially with poor initial placement

**Why?** Better initial centroids are closer to final positions, requiring fewer updates.

#### 3. **Consistency and Stability**

- **K-Means++**: More consistent results across multiple runs
- **Random**: High variance - different runs can produce very different results

**Why?** K-Means++ uses a deterministic selection process (probabilistic but guided), while random is purely stochastic.

#### 4. **Computational Cost**

- **K-Means++**: Slightly slower initialization (O(nkd) vs O(kd))
- **Random**: Fastest initialization

**Trade-off**: The extra initialization time is usually offset by faster convergence, resulting in similar or better total runtime.

### Visual Comparison

When visualized, you'll typically see:
- **K-Means++**: Centroids are well-distributed, clusters are more balanced
- **Random**: Centroids may cluster together, leading to imbalanced or merged clusters

### When to Use Each Method

| Method | Best For | Avoid When |
|--------|----------|------------|
| **Random** | Quick experiments, very large datasets where initialization time matters | Production systems, when quality matters |
| **Naive Sharding** | Data with clear linear separation along one dimension | Complex cluster shapes, non-linear patterns |
| **K-Means++** | Production systems, when quality and consistency matter | Extremely large datasets where initialization overhead is critical |

---

## Implementation Details

### Class Structure

The `KMeans` class implements:

- **Initialization methods**: `_initialize_random()`, `_initialize_naive_sharding()`, `_initialize_kmeans_plusplus()`
- **Core algorithm**: `fit()` method that iterates assignment and update steps
- **Prediction**: `predict()` method for new data points
- **Metrics**: Inertia calculation and convergence tracking

### Key Functions

1. **`compare_initialization_methods()`**: Runs K-Means with all three methods and compares results
2. **`visualize_clustering()`**: Creates side-by-side visualizations of clustering results
3. **`plot_convergence_comparison()`**: Shows how inertia decreases over iterations for each method
4. **`generate_sample_datasets()`**: Creates test datasets with different characteristics

### Algorithm Complexity

- **Time Complexity**: O(nkdi) where:
  - n = number of data points
  - k = number of clusters
  - d = number of dimensions
  - i = number of iterations

- **Space Complexity**: O(nk + kd) for storing labels and centroids

---

## Usage

### Basic Usage

```python
from cluster import KMeans
import numpy as np

# Generate or load your data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Create and fit K-Means model with K-Means++ initialization
kmeans = KMeans(n_clusters=2, init_method='kmeans++', random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Get centroids
centroids = kmeans.centroids

# Predict clusters for new data
new_data = np.array([[2, 3], [11, 3]])
predictions = kmeans.predict(new_data)
```

### Running the Full Comparison

```python
from cluster import main

# Run complete comparison with visualizations
main()
```

This will:
1. Generate sample datasets
2. Run K-Means with all three initialization methods
3. Display visualizations
4. Print comparison metrics

### Custom Dataset Comparison

```python
from cluster import compare_initialization_methods, visualize_clustering
import numpy as np

# Your custom data
X = np.random.rand(100, 2)  # 100 points, 2 features

# Compare methods
results = compare_initialization_methods(X, n_clusters=3, random_state=42)

# Visualize
visualize_clustering(X, results, title="My Dataset Comparison")
```

---

## Results and Analysis

### Expected Results

When running the comparison, you should observe:

1. **Inertia Values**: K-Means++ typically achieves the lowest inertia
2. **Iteration Count**: K-Means++ converges in fewer iterations
3. **Visual Quality**: K-Means++ produces more balanced and well-separated clusters
4. **Consistency**: K-Means++ shows less variation across runs

### Sample Output

```
COMPARING INITIALIZATION METHODS
======================================================================

RANDOM Initialization:
--------------------------------------------------
  Inertia (WCSS): 245.32
  Iterations: 12
  Time: 0.0234 seconds

NAIVE_SHARDING Initialization:
--------------------------------------------------
  Inertia (WCSS): 198.45
  Iterations: 9
  Time: 0.0198 seconds

KMEANS++ Initialization:
--------------------------------------------------
  Inertia (WCSS): 187.23
  Iterations: 7
  Time: 0.0212 seconds

SUMMARY COMPARISON
--------------------------------------------------
Method               Inertia         Iterations      Time (s)        
--------------------------------------------------
random               245.3200        12              0.0234          
naive_sharding       198.4500        9               0.0198          
kmeans++             187.2300        7               0.0212          

K-MEANS++ vs RANDOM INITIALIZATION
--------------------------------------------------
Inertia Improvement: 23.70% (Lower is better)
Iteration Reduction: 41.67% (K-Means++ converged faster)
Time Difference: -0.0022 seconds (faster for K-Means++)
```

### Interpretation

- **Inertia Improvement**: Shows how much better K-Means++ clusters are
- **Iteration Reduction**: Demonstrates faster convergence
- **Time Difference**: Shows total runtime comparison (initialization + iterations)

---

## Conclusion

### Key Findings

1. **K-Means++ is Superior**: Consistently produces better cluster quality with lower inertia
2. **Faster Convergence**: Requires fewer iterations to reach convergence
3. **More Stable**: Produces consistent results across multiple runs
4. **Worth the Trade-off**: Slightly slower initialization is offset by faster convergence

### Recommendations

- **Use K-Means++** for production systems and when cluster quality matters
- **Use Random** only for quick experiments or when initialization time is critical
- **Use Naive Sharding** when data has clear linear separation along one dimension

### Theoretical Background

K-Means++ was proposed by Arthur and Vassilvitskii (2007) and provides an O(log k) approximation guarantee compared to the optimal solution. This makes it the standard initialization method in modern implementations (e.g., scikit-learn's default).

### Future Extensions

Potential improvements and extensions:
- Multiple random initializations with best result selection
- Adaptive k selection (elbow method, silhouette score)
- Handling of empty clusters
- Support for different distance metrics (Manhattan, cosine, etc.)
- Mini-batch K-Means for large datasets

---

## Dependencies

- `numpy` - Numerical computations
- `matplotlib` - Visualizations
- `scikit-learn` - For generating sample datasets (optional, can use custom data)

## Installation

```bash
# If using virtual environment (recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install numpy matplotlib scikit-learn
```

## Author

Implementation for AI & ML Activity 1 - K-Means Clustering Investigation

## References

1. Arthur, D., & Vassilvitskii, S. (2007). "k-means++: The advantages of careful seeding." Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms.
2. MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations." Proceedings of the fifth Berkeley symposium on mathematical statistics and probability.
3. Lloyd, S. P. (1982). "Least squares quantization in PCM." IEEE transactions on information theory.

---

## License

This project is for educational purposes.

