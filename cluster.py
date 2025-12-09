import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
import time
from typing import Tuple, List, Optional
import random


class KMeans:
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 300, 
                 init_method: str = 'kmeans++', random_state: Optional[int] = None,
                 tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init_method = init_method
        self.random_state = random_state
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.convergence_history = []
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)
    
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def _initialize_random(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        # Randomly select k indices
        indices = np.random.choice(n_samples, size=self.n_clusters, replace=False)
        centroids = X[indices].copy()
        return centroids
    
    def _initialize_naive_sharding(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        sorted_indices = np.argsort(X[:, 0])
        sorted_X = X[sorted_indices]
        shard_size = n_samples // self.n_clusters
        centroids = []
        
        for i in range(self.n_clusters):
            start_idx = i * shard_size
            if i == self.n_clusters - 1:
                end_idx = n_samples
            else:
                end_idx = (i + 1) * shard_size
            
            shard = sorted_X[start_idx:end_idx]
            centroid = np.mean(shard, axis=0)
            centroids.append(centroid)
        
        return np.array(centroids)
    
    def _initialize_kmeans_plusplus(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Step 1: Choose first centroid randomly
        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx].copy()
        
        # Step 2: Choose remaining centroids
        for k in range(1, self.n_clusters):
            # Calculate distances from each point to nearest centroid
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                # Find minimum distance to any existing centroid
                min_dist = float('inf')
                for j in range(k):
                    dist = self._euclidean_distance(X[i], centroids[j])
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist ** 2  # Squared distance
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / np.sum(distances)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.random()
            
            # Find the index where cumulative probability exceeds r
            next_idx = np.searchsorted(cumulative_probs, r)
            centroids[k] = X[next_idx].copy()
        
        return centroids
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids based on the selected method."""
        if self.init_method == 'random':
            return self._initialize_random(X)
        elif self.init_method == 'naive_sharding':
            return self._initialize_naive_sharding(X)
        elif self.init_method == 'kmeans++':
            return self._initialize_kmeans_plusplus(X)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
    
    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each data point to the nearest centroid."""
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = [self._euclidean_distance(X[i], centroid) 
                        for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids to be the mean of points in each cluster."""
        new_centroids = np.zeros_like(self.centroids)
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, keep the old centroid
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia)."""
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                for point in cluster_points:
                    inertia += self._euclidean_distance(point, self.centroids[k]) ** 2
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            inertia = self._calculate_inertia(X, self.labels)
            self.convergence_history.append(inertia)
            new_centroids = self._update_centroids(X, self.labels)
            
            centroid_shift = np.sum([self._euclidean_distance(
                self.centroids[k], new_centroids[k]) 
                for k in range(self.n_clusters)])
            
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        self.inertia_ = self._calculate_inertia(X, self.labels)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster for each sample in X."""
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")
        return self._assign_clusters(X)


def compare_initialization_methods(X: np.ndarray, n_clusters: int = 3, 
                                   random_state: int = 42, 
                                   show_centroids: bool = True) -> dict:
    methods = ['random', 'naive_sharding', 'kmeans++']
    results = {}
    
    print("\n" + "="*70)
    print("COMPARING INITIALIZATION METHODS")
    print("="*70)
    
    method_seeds = {
        'random': random_state,
        'naive_sharding': random_state,
        'kmeans++': random_state + 1
    }
    
    for method in methods:
        print(f"\n{method.upper()} Initialization:")
        print("-" * 50)
        start_time = time.time()
        kmeans = KMeans(n_clusters=n_clusters, init_method=method, 
                       random_state=method_seeds[method])
        kmeans.fit(X)
        elapsed_time = time.time() - start_time
        inertia = kmeans.inertia_
        n_iterations = kmeans.n_iter_
        
        results[method] = {
            'model': kmeans,
            'inertia': inertia,
            'iterations': n_iterations,
            'time': elapsed_time,
            'convergence_history': kmeans.convergence_history
        }
        
        print(f"  Inertia (WCSS): {inertia:.4f}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Time: {elapsed_time:.4f} seconds")
        if show_centroids:
            print(f"  Final centroids:\n{kmeans.centroids}")
    
    return results


def visualize_clustering(X: np.ndarray, results: dict, title: str = "K-Means Clustering Comparison"):
    n_methods = len(results)
    fig, axes = plt.subplots(1, n_methods, figsize=(15, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(results['random']['model'].centroids))))
    
    for idx, (method, result) in enumerate(results.items()):
        ax = axes[idx]
        model = result['model']
        labels = model.labels
        centroids = model.centroids
        
        # Plot data points
        for k in range(model.n_clusters):
            cluster_points = X[labels == k]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=[colors[k]], label=f'Cluster {k}', alpha=0.6, s=50)
        
        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='red', marker='x', s=200, linewidths=3, 
                  label='Centroids', zorder=10)
        
        ax.set_title(f'{method.upper()}\nInertia: {result["inertia"]:.2f}, '
                    f'Iterations: {result["iterations"]}', fontsize=10)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_convergence_comparison(results: dict):
    plt.figure(figsize=(10, 6))
    
    for method, result in results.items():
        history = result['convergence_history']
        plt.plot(history, label=method.upper(), marker='o', markersize=4)
    
    plt.xlabel('Iteration')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Convergence Comparison: Inertia vs Iterations', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def generate_sample_datasets():
    """
    Generate sample datasets for testing.
    
    Returns:
    --------
    dict : Dictionary of datasets with different characteristics
    """
    datasets = {}
    
    # Dataset 1: Well-separated blobs (3 clusters)
    X1, y1 = make_blobs(n_samples=300, centers=3, n_features=2, 
                       random_state=42, cluster_std=0.60)
    datasets['well_separated'] = X1
    
    # Dataset 2: Overlapping clusters (more challenging)
    X2, y2 = make_blobs(n_samples=300, centers=3, n_features=2, 
                       random_state=42, cluster_std=2.0)
    datasets['overlapping'] = X2
    
    # Dataset 3: Uneven cluster sizes and densities
    # When n_samples is a list, centers should be None (inferred from list length)
    X3, y3 = make_blobs(n_samples=[100, 150, 50], centers=None, n_features=2, 
                       random_state=42, cluster_std=[0.8, 1.5, 1.2])
    datasets['uneven_sizes'] = X3
    
    # Dataset 4: Clusters with different shapes (more challenging)
    X4, y4 = make_blobs(n_samples=300, centers=3, n_features=2, 
                       random_state=42, cluster_std=[1.2, 0.8, 1.5])
    datasets['varying_density'] = X4
    
    return datasets


def demonstrate_variance(X: np.ndarray, n_clusters: int = 3, n_trials: int = 10):
    """
    Demonstrate variance in results across multiple runs.
    This shows why K-Means++ is more stable than random initialization.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    n_clusters : int
        Number of clusters
    n_trials : int
        Number of trials to run
    """
    print("\n" + "="*70)
    print("VARIANCE ANALYSIS: Multiple Runs")
    print("="*70)
    print(f"Running {n_trials} trials for each method...\n")
    
    random_inertias = []
    kmeanspp_inertias = []
    
    for trial in range(n_trials):
        # Random initialization
        kmeans_random = KMeans(n_clusters=n_clusters, init_method='random',
                              random_state=42 + trial * 10)
        kmeans_random.fit(X)
        random_inertias.append(kmeans_random.inertia_)
        
        # K-Means++ initialization
        kmeans_pp = KMeans(n_clusters=n_clusters, init_method='kmeans++',
                          random_state=42 + trial * 10)
        kmeans_pp.fit(X)
        kmeanspp_inertias.append(kmeans_pp.inertia_)
    
    print(f"{'Method':<20} {'Mean Inertia':<20} {'Std Dev':<20} {'Min':<15} {'Max':<15}")
    print("-" * 90)
    print(f"{'Random':<20} {np.mean(random_inertias):<20.4f} "
          f"{np.std(random_inertias):<20.4f} {np.min(random_inertias):<15.4f} "
          f"{np.max(random_inertias):<15.4f}")
    print(f"{'K-Means++':<20} {np.mean(kmeanspp_inertias):<20.4f} "
          f"{np.std(kmeanspp_inertias):<20.4f} {np.min(kmeanspp_inertias):<15.4f} "
          f"{np.max(kmeanspp_inertias):<15.4f}")
    
    variance_reduction = (1 - np.std(kmeanspp_inertias) / np.std(random_inertias)) * 100
    print(f"\nK-Means++ reduces variance by {variance_reduction:.2f}% compared to Random")
    print("="*70)


def main():
    """
    Main function to demonstrate K-Means with different initialization methods.
    """
    print("\n" + "="*70)
    print("K-MEANS CLUSTERING: INITIALIZATION METHODS COMPARISON")
    print("="*70)
    
    # Generate sample datasets
    datasets = generate_sample_datasets()
    
    # Test on each dataset
    for dataset_name, X in datasets.items():
        print(f"\n\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Compare initialization methods
        results = compare_initialization_methods(X, n_clusters=3, random_state=42)
        
        # Visualize results
        visualize_clustering(X, results, 
                           title=f"K-Means Comparison: {dataset_name.replace('_', ' ').title()}")
        
        # Plot convergence comparison
        plot_convergence_comparison(results)
        
        # Print summary comparison
        print("\n" + "-"*70)
        print("SUMMARY COMPARISON")
        print("-"*70)
        print(f"{'Method':<20} {'Inertia':<15} {'Iterations':<15} {'Time (s)':<15}")
        print("-"*70)
        for method, result in results.items():
            print(f"{method:<20} {result['inertia']:<15.4f} "
                  f"{result['iterations']:<15} {result['time']:<15.4f}")
        
        # K-Means++ vs Random comparison
        print("\n" + "-"*70)
        print("K-MEANS++ vs RANDOM INITIALIZATION")
        print("-"*70)
        random_result = results['random']
        kmeanspp_result = results['kmeans++']
        
        inertia_improvement = ((random_result['inertia'] - kmeanspp_result['inertia']) 
                              / random_result['inertia'] * 100)
        iter_improvement = ((random_result['iterations'] - kmeanspp_result['iterations']) 
                           / random_result['iterations'] * 100)
        time_diff = kmeanspp_result['time'] - random_result['time']
        
        print(f"Inertia Improvement: {inertia_improvement:.2f}% "
              f"(Lower is better - K-Means++ achieved {inertia_improvement:.2f}% lower inertia)")
        print(f"Iteration Reduction: {iter_improvement:.2f}% "
              f"(K-Means++ converged in {iter_improvement:.2f}% fewer iterations)")
        print(f"Time Difference: {time_diff:.4f} seconds "
              f"({'slower' if time_diff > 0 else 'faster'} for K-Means++)")
        
        # Note about well-separated data
        if dataset_name == 'well_separated' and abs(inertia_improvement) < 1.0:
            print("\n  Note: For well-separated clusters, all methods often converge")
            print("  to the same optimal solution. Differences become more apparent")
            print("  with overlapping or complex cluster structures.")
    
    # Demonstrate variance on the overlapping dataset (most challenging)
    if 'overlapping' in datasets:
        demonstrate_variance(datasets['overlapping'], n_clusters=3, n_trials=10)
if __name__ == "__main__":
    main()

