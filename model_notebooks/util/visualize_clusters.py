from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def visualize_clusters(data, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', s=10, alpha=0.8)

    # Highlight noise
    if -1 in labels:
        noise_mask = np.array(labels) == -1
        plt.scatter(reduced[noise_mask, 0], reduced[noise_mask, 1], c='red', s=10, label='Noise', alpha=0.6)

    plt.title('Clusters (PCA Reduced)')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(loc="best", markerscale=2, fontsize='small', frameon=True)
    plt.tight_layout()
    plt.show()