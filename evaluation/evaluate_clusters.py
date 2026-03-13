from sklearn.metrics import silhouette_score


def evaluate_clusters(data, clusters):

    score = silhouette_score(data, clusters)

    print("\nSilhouette Score:", score)