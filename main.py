print("Customer Segmentation Pipeline Starting...\n")

from evaluation.customer_personas import describe_clusters
from data.generate_data import generate_customer_data
from preprocessing.preprocess import preprocess_data
from models.train_kmeans import train_kmeans
from visualization.visualize_clusters import visualize_clusters
from evaluation.find_optimal_clusters import find_optimal_clusters
from evaluation.evaluate_clusters import evaluate_clusters


def main():

    print("Step 1: Generating synthetic customer data")
    df = generate_customer_data()

    print("Step 2: Preprocessing data")
    scaled_data = preprocess_data(df)

    print("Step 3: Finding optimal number of clusters")
    best_k = find_optimal_clusters(scaled_data)

    print("Step 4: Training clustering model")
    model, clusters = train_kmeans(scaled_data, best_k)

    print("Step 5: Evaluating clusters")
    evaluate_clusters(scaled_data, clusters)

    print("Step 6: Visualizing segments")
    visualize_clusters(scaled_data, clusters)

    df["Cluster"] = clusters

    print("\nCustomer Segment Summary:")
    print(df.groupby("Cluster").mean())

    describe_clusters(df)

    df.to_csv("segmented_customers.csv", index=False)

    print("\nSegmented dataset saved as segmented_customers.csv")


if __name__ == "__main__":
    main()