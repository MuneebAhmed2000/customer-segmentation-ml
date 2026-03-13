from sklearn.cluster import KMeans


def train_kmeans(data, n_clusters=4):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    clusters = kmeans.fit_predict(data)

    return kmeans, clusters