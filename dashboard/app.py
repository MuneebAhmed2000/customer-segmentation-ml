import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

from data.generate_data import generate_customer_data
from preprocessing.preprocess import preprocess_data
from models.train_kmeans import train_kmeans
from evaluation.find_optimal_clusters import find_optimal_clusters


st.set_page_config(page_title="Customer Segmentation ML", layout="wide")

st.title("Customer Segmentation Machine Learning Dashboard")

st.write(
"""
This dashboard demonstrates an end-to-end **Machine Learning Customer Segmentation Pipeline** using synthetic data.

Pipeline:
- Data Generation
- Feature Scaling
- Optimal Cluster Selection
- KMeans Clustering
- Visualization & Personas
"""
)

if st.button("Run Customer Segmentation Pipeline"):

    # ===============================
    # GENERATE DATA
    # ===============================

    df = generate_customer_data()

    st.subheader("Sample Customer Data")
    st.dataframe(df.head())

    # ===============================
    # PREPROCESS DATA
    # ===============================

    scaled_data = preprocess_data(df)

    # ===============================
    # FIND OPTIMAL CLUSTERS
    # ===============================

    best_k = find_optimal_clusters(scaled_data)

    st.success(f"Optimal number of clusters: {best_k}")

    # ===============================
    # TRAIN MODEL
    # ===============================

    model, clusters = train_kmeans(scaled_data, best_k)

    df["Cluster"] = clusters

    st.subheader("Clustered Customers")
    st.dataframe(df.head())

    # ===============================
    # CLUSTER SUMMARY
    # ===============================

    st.subheader("Cluster Summary Statistics")

    summary = df.groupby("Cluster").mean()
    st.dataframe(summary)

    # ===============================
    # KPI DASHBOARD METRICS
    # ===============================

    st.subheader("Customer Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(df))
    col2.metric("Average Income", f"${df['Annual_Income'].mean():,.0f}")
    col3.metric("Average Spending Score", f"{df['Spending_Score'].mean():.1f}")
    col4.metric("Segments Found", best_k)

    # ===============================
    # INTERACTIVE CLUSTER VISUALIZATION
    # ===============================

    st.subheader("Interactive Cluster Visualization")

    fig = px.scatter(
        df,
        x="Annual_Income",
        y="Spending_Score",
        color="Cluster",
        size="Purchase_Frequency",
        hover_data=["Age", "Online_Visits"],
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # PCA VISUALIZATION
    # ===============================

    st.subheader("PCA Cluster Visualization")

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(
        components,
        columns=["PCA1", "PCA2"]
    )

    pca_df["Cluster"] = clusters

    fig2 = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        title="Clusters Visualized Using PCA"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ===============================
    # 3D CLUSTER VISUALIZATION
    # ===============================

    st.subheader("3D Customer Segmentation")

    fig3d = px.scatter_3d(
        df,
        x="Annual_Income",
        y="Spending_Score",
        z="Purchase_Frequency",
        color="Cluster",
        size="Online_Visits",
        hover_data=["Age"],
        title="3D Customer Segmentation"
    )

    st.plotly_chart(fig3d, use_container_width=True)

    # ===============================
    # CUSTOMER PERSONAS
    # ===============================

    st.subheader("Customer Personas")

    for cluster in summary.index:

        income = summary.loc[cluster, "Annual_Income"]
        spending = summary.loc[cluster, "Spending_Score"]
        freq = summary.loc[cluster, "Purchase_Frequency"]

        if income > 100000:
            persona = "Luxury High-Value Customers"

        elif freq > 20:
            persona = "Loyal Repeat Buyers"

        elif spending < 40:
            persona = "Window Shoppers"

        else:
            persona = "Discount Driven Shoppers"

        st.info(f"Cluster {cluster}: {persona}")

    # ===============================
    # AUTOMATIC SEGMENT INSIGHTS
    # ===============================

    st.subheader("Segment Insights")

    highest_income = summary["Annual_Income"].idxmax()
    highest_spending = summary["Spending_Score"].idxmax()
    most_frequent = summary["Purchase_Frequency"].idxmax()

    st.success(f"Cluster {highest_income} has the highest average income.")

    st.success(f"Cluster {highest_spending} spends the most on average.")

    st.success(f"Cluster {most_frequent} purchases most frequently.")

    # ===============================
    # EXPORT DATA
    # ===============================

    st.subheader("Export Segmented Data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Segmented Customer Dataset",
        data=csv,
        file_name="segmented_customers.csv",
        mime="text/csv",
    )