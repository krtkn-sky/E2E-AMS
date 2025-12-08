import os
import pandas as pd

# ----- PATHS -----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "events_clean.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "data", "events_features.csv")

print("Loading:", CLEAN_PATH)

# Load data
df = pd.read_csv(CLEAN_PATH)

# ----- 1. TIME FEATURES -----
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.weekday

# ----- 2. ENCODING EVENT TYPE -----
df = pd.get_dummies(df, columns=["event"], prefix="evt")

# ----- 3. USER BEHAVIOR AGGREGATES -----
user_stats = df.groupby("user_id").agg({
    "interaction_score": ["sum", "mean"],
    "product_id": "nunique"
})

user_stats.columns = ["user_total_score", "user_avg_score", "unique_products_viewed"]
user_stats.reset_index(inplace=True)

df = df.merge(user_stats, on="user_id", how="left")

# ----- 4. PRODUCT POPULARITY FEATURE -----
product_views = df.groupby("product_id")["evt_view"].sum().rename("product_view_count")
df = df.merge(product_views, on="product_id", how="left")

# ----- 5. OPTIONAL: PURCHASE LABEL -----
df["label_purchase"] = (df["evt_purchase"] > 0).astype(int)

# Save engineered dataset
df.to_csv(FEATURE_PATH, index=False)
print("Feature engineering completed.")
print("Saved:", FEATURE_PATH)
print("Rows:", len(df))
print("Columns:", len(df.columns))
