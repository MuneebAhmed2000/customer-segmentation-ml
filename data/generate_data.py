import pandas as pd
import numpy as np


def generate_customer_data(n_customers=500):

    np.random.seed(42)

    data = pd.DataFrame({
        "Age": np.random.randint(18, 70, n_customers),
        "Annual_Income": np.random.randint(20000, 150000, n_customers),
        "Spending_Score": np.random.randint(1, 100, n_customers),
        "Purchase_Frequency": np.random.randint(1, 30, n_customers),
        "Online_Visits": np.random.randint(1, 50, n_customers)
    })

    return data