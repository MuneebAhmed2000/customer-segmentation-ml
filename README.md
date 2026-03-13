# Customer Segmentation Machine Learning Pipeline

An end-to-end machine learning pipeline that automatically segments customers using unsupervised learning techniques.

This project demonstrates how data science can be used to identify meaningful customer segments to improve marketing strategies and business decision making.

---

# Project Overview

Customer segmentation helps companies understand different types of customers based on behavior and spending patterns.

This project builds a full pipeline that:

• Generates synthetic customer data
• Preprocesses and scales features
• Finds the optimal number of clusters automatically
• Applies K-Means clustering
• Visualizes customer segments
• Generates customer personas
• Produces interactive analytics dashboards

---

# Features

### Machine Learning Pipeline

* Synthetic dataset generation
* Data preprocessing
* Optimal cluster detection using silhouette score
* K-Means clustering

### Interactive Dashboard

Built using Streamlit.

Includes:

* Interactive cluster visualizations
* PCA dimensionality reduction
* 3D segmentation visualization
* Customer persona generation
* Automatic cluster insights
* Downloadable segmented dataset

---

# Technologies Used

Python
Pandas
Scikit-Learn
Plotly
Streamlit
NumPy

---

# Project Structure

customer_segmentation_ml

data/
 generate_data.py

preprocessing/
 preprocess.py

models/
 train_kmeans.py

evaluation/
 find_optimal_clusters.py
 evaluate_clusters.py
 customer_personas.py

visualization/
 visualize_clusters.py

dashboard/
 app.py

main.py
requirements.txt

---

# Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run the dashboard:

```
python -m streamlit run dashboard/app.py
```

---

# Example Dashboard Capabilities

The dashboard allows users to:

• Generate customer datasets
• Automatically detect optimal cluster count
• Visualize segments in 2D, PCA, and 3D space
• View cluster statistics and customer personas
• Export segmented customer data

---

# Future Improvements

• Real customer datasets
• Marketing campaign optimization
• Predictive lifetime value modeling
• Deployment as a cloud analytics tool

---

# Author
Muneeb Ahmed 

Data Science Portfolio Project
