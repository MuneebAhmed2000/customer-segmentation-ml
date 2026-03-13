import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd


def visualize_clusters(data, clusters):

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    df = pd.DataFrame({
        "PC1": reduced_data[:, 0],
        "PC2": reduced_data[:, 1],
        "Cluster": clusters
    })

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        data=df
    )

    plt.title("Customer Segments")

    plt.show()