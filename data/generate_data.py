import pandas as pd
import numpy as np


def generate_customer_data():

    np.random.seed(42)

    n = 150

    # Luxury shoppers
    luxury = pd.DataFrame({
        "Age": np.random.normal(45, 8, n),
        "Annual_Income": np.random.normal(120000, 15000, n),
        "Spending_Score": np.random.normal(85, 10, n),
        "Purchase_Frequency": np.random.normal(22, 5, n),
        "Online_Visits": np.random.normal(15, 4, n)
    })

    # Bargain hunters
    bargain = pd.DataFrame({
        "Age": np.random.normal(30, 7, n),
        "Annual_Income": np.random.normal(40000, 10000, n),
        "Spending_Score": np.random.normal(70, 10, n),
        "Purchase_Frequency": np.random.normal(18, 5, n),
        "Online_Visits": np.random.normal(30, 8, n)
    })

    # Loyal regulars
    loyal = pd.DataFrame({
        "Age": np.random.normal(40, 6, n),
        "Annual_Income": np.random.normal(80000, 12000, n),
        "Spending_Score": np.random.normal(60, 10, n),
        "Purchase_Frequency": np.random.normal(25, 4, n),
        "Online_Visits": np.random.normal(20, 6, n)
    })

    # Window shoppers
    browsers = pd.DataFrame({
        "Age": np.random.normal(27, 5, n),
        "Annual_Income": np.random.normal(50000, 9000, n),
        "Spending_Score": np.random.normal(30, 10, n),
        "Purchase_Frequency": np.random.normal(6, 3, n),
        "Online_Visits": np.random.normal(35, 7, n)
    })

    df = pd.concat([luxury, bargain, loyal, browsers], ignore_index=True)

    df = df.clip(lower=0)

    return df