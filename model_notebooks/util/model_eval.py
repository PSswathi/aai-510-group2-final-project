from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


def evaluate_clustering(data, labels, show_plot=True):
    """
    Evaluates a clustering model using silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    Parameters:
        model: fitted clustering model
        data: scaled feature matrix
        labels: cluster labels

    Returns:
        dict: Evaluation scores
    """
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = np.sum(np.array(labels) == -1) if -1 in labels else 0

    results = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
    }

    if n_clusters > 1:
        try:
            results["silhouette_score"] = silhouette_score(data, labels)
        except:
            results["silhouette_score"] = None
        try:
            results["davies_bouldin_score"] = davies_bouldin_score(data, labels)
        except:
            results["davies_bouldin_score"] = None
        try:
            results["calinski_harabasz_score"] = calinski_harabasz_score(data, labels)
        except:
            results["calinski_harabasz_score"] = None
    else:
        results["silhouette_score"] = None
        results["davies_bouldin_score"] = None
        results["calinski_harabasz_score"] = None

    if show_plot:
        results_df = pd.DataFrame(results.items(), columns=["Metric", "Value"])
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis('off')
        table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        plt.tight_layout()
        plt.show()

    return results
