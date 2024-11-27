import matplotlib.pyplot as plt
import os


def create_cluster_plot(numeric_data, cluster_labels, save_path="./static/cluster_visualization.png"):
    """Create a scatter plot of clusters and save it."""
    plt.figure(figsize=(8, 6))

    # Scatter plot of the first two numeric features
    for cluster in set(cluster_labels):
        cluster_points = numeric_data[cluster_labels == cluster]
        plt.scatter(
            cluster_points.iloc[:, 0],
            cluster_points.iloc[:, 1],
            label=f"Cluster {cluster}",
            s=30,
            alpha=0.7,
        )

    plt.title("Cluster Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    os.makedirs("./static", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    return save_path
