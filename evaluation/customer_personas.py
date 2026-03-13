def describe_clusters(df):

    summary = df.groupby("Cluster").mean()

    print("\nCustomer Personas\n")

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

        print(f"Cluster {cluster}: {persona}")