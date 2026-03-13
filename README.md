# Customer Segmentation Machine Learning Pipeline

Live Demo: https://muneebs-customer-segmentation-ml.streamlit.app

## Overview

This project demonstrates an end-to-end **Machine Learning customer segmentation pipeline**.
It generates synthetic customer data, preprocesses features, determines the optimal number of clusters, and segments customers using **K-Means clustering**.

An interactive dashboard built with **Streamlit** allows users to explore the clusters through multiple visualizations and automatically generated business insights.

This project simulates how data science teams segment customers to improve marketing, personalization, and product strategy.

---

## Live Dashboard

Try the interactive demo:

**https://muneebs-customer-segmentation-ml.streamlit.app**

The dashboard allows users to:

* Generate synthetic customer data
* Run the full ML pipeline
* Visualize clusters interactively
* Explore 3D segmentation
* View automated customer personas
* Export segmented datasets

---

## Project Architecture

Data Pipeline:

Data Generation
→ Feature Scaling
→ Optimal Cluster Selection
→ K-Means Model Training
→ Dimensionality Reduction (PCA)
→ Visualization & Insights

---

## Features

### Machine Learning Pipeline

* Synthetic dataset generation
* Feature preprocessing and scaling
* Optimal cluster selection using the Elbow Method
* Customer segmentation with K-Means clustering

### Interactive Dashboard

* Interactive cluster visualization
* PCA dimensionality reduction
* 3D customer segmentation
* KPI metric dashboard
* Downloadable segmented dataset

### Automated Business Insights

The dashboard automatically generates customer personas and identifies:

* High income segments
* High spending customers
* Frequent buyers
* Discount-driven shoppers

---

## Example Dashboard Output

Add screenshots of the dashboard here.

Example:

screenshots/dashboard_overview.png
screenshots/cluster_visualization.png
screenshots/pca_visualization.png

---

## Project Structure

customer-segmentation-ml/

data/
 generate_data.py

preprocessing/
 preprocess.py

models/
 train_kmeans.py

evaluation/
 find_optimal_clusters.py
 customer_personas.py

dashboard/
 app.py

requirements.txt
README.md

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Matplotlib

---

## Installation

Clone the repository:

git clone https://github.com/MuneebAhmed2000/customer-segmentation-ml.git

Install dependencies:

pip install -r requirements.txt

Run the dashboard:

streamlit run dashboard/app.py

---

## Machine Learning Approach

Customer segmentation is performed using **K-Means clustering**.

Steps:

1. Generate synthetic customer data
2. Standardize features
3. Determine optimal cluster count using the Elbow Method
4. Train K-Means clustering model
5. Visualize clusters using PCA and 3D projections

---

## Business Applications

Customer segmentation is widely used in:

* Marketing personalization
* Customer lifetime value analysis
* Product recommendation systems
* Pricing strategy
* Customer retention programs

---

## Future Improvements

* Real dataset integration
* Advanced clustering models (DBSCAN, Hierarchical Clustering)
* Customer lifetime value prediction
* Automated cluster reporting

---

## Author

Muneeb Ahmed
Data Science & Machine Learning Portfolio Project
