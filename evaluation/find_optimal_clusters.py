from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def find_optimal_clusters(data):

    best_k = 2
    best_score = -1

    for k in range(2, 10):

        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)

        score = silhouette_score(data, labels)

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\nBest number of clusters: {best_k}")
    print(f"Best silhouette score: {best_score}")

    return best_k