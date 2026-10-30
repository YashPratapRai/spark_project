
import pandas as pd
import numpy as np

# Load generated predictions
df = pd.read_csv("outputs/predictions.csv")

# Calculate prediction errors
df["error"] = df["actual_revenue"] - df["predicted_revenue"]
df["absolute_error"] = df["error"].abs()
df["squared_error"] = df["error"] ** 2

print("\n========== CURRENT MODEL ERROR ANALYSIS ==========")

print("\nTotal Prediction Records:", len(df))

print("\n========== ERROR STATISTICS ==========")
print(df["absolute_error"].describe())

print("\n========== RE-CALCULATED METRICS ==========")
mae = df["absolute_error"].mean()
rmse = np.sqrt(df["squared_error"].mean())
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

print("\n========== TOP 15 HIGH-ERROR ORDERS ==========")
worst = df.sort_values(
    by="absolute_error", ascending=False
).head(15)

print(worst[
    ["order_id", "category", "actual_revenue",
     "predicted_revenue", "absolute_error"]
].to_string(index=False))

print("\n========== ERROR BY CATEGORY ==========")
category_errors = df.groupby("category").agg(
    Average_Absolute_Error=("absolute_error", "mean"),
    Maximum_Error=("absolute_error", "max"),
    Records=("absolute_error", "count")
).round(2)

print(category_errors)

print("\n========== ERROR PERCENTILES ==========")
for percentile in [50, 75, 90, 95, 99]:
    value = np.percentile(df["absolute_error"], percentile)
    print(f"{percentile}th percentile: ${value:.2f}")

print("\nError analysis completed!")
