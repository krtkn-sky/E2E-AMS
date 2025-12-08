import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PATH FIX ----------------
# Get project root directory (E2E-AMS)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CLEAN_PATH = os.path.join(BASE_DIR, "data", "events_clean.csv")

print("Loading:", CLEAN_PATH)

# Load the cleaned data
df = pd.read_csv(CLEAN_PATH)

# -------- 1. BASIC OVERVIEW --------
print("\n--- DATA OVERVIEW ---")
print(df.head())
print(df.info())
print(df.describe())

# -------- 2. EVENT DISTRIBUTION --------
event_counts = df["event"].value_counts()
print("\n--- EVENT DISTRIBUTION ---")
print(event_counts)

plt.figure(figsize=(7,5))
event_counts.plot(kind="bar")
plt.title("Event Distribution")
plt.xlabel("Event Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "analysis", "event_distribution.png"))
plt.close()

# -------- 3. USER ACTIVITY --------
user_activity = df["user_id"].value_counts()
print("\n--- TOP ACTIVE USERS ---")
print(user_activity.head())

plt.figure(figsize=(7,5))
user_activity.head(10).plot(kind="bar")
plt.title("Top 10 Most Active Users")
plt.xlabel("User ID")
plt.ylabel("Activity Count")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "analysis", "top_users.png"))
plt.close()

# -------- 4. PRODUCT POPULARITY --------
product_views = df[df["event"] == "view"]["product_id"].value_counts()
product_purchases = df[df["event"] == "purchase"]["product_id"].value_counts()

product_summary = pd.DataFrame({
    "views": product_views,
    "purchases": product_purchases
}).fillna(0).astype(int)

product_summary_path = os.path.join(BASE_DIR, "analysis", "product_summary.csv")
product_summary.to_csv(product_summary_path)

print("\n--- PRODUCT SUMMARY (views vs purchases) saved ---")

plt.figure(figsize=(7,5))
product_summary["views"].head(10).plot(kind="bar")
plt.title("Top 10 Viewed Products")
plt.xlabel("Product ID")
plt.ylabel("Views")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "analysis", "top_viewed_products.png"))
plt.close()

# -------- 5. TIME-BASED ANALYSIS --------
hourly = df.groupby("hour")["interaction_score"].count()

plt.figure(figsize=(7,5))
hourly.plot(kind="line")
plt.title("Activity by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("Events")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "analysis", "activity_by_hour.png"))
plt.close()

print("\nEDA completed. Charts saved in analysis/ folder.\n")
