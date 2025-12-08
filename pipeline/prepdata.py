import pandas as pd

INPUT_PATH = "/home/ak/Desktop/projects/E2E-AMS/data/events_features.csv"
OUTPUT_PATH = "/home/ak/Desktop/projects/E2E-AMS/data/events_features_dashboard.csv"

def prepare_dashboard_csv():
    df = pd.read_csv(INPUT_PATH)

    cols = [
        "user_id", "product_id",
        "hour", "day_of_week", "is_weekend", "interaction_score",
        "day", "weekday",
        "evt_add_to_cart", "evt_click", "evt_view",
        "user_total_score", "user_avg_score",
        "unique_products_viewed", "product_view_count"
    ]

    df_clean = df[cols]

    bool_cols = ["evt_add_to_cart", "evt_click", "evt_view"]
    for col in bool_cols:
        df_clean[col] = df_clean[col].astype(int)

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated dashboard CSV at: {OUTPUT_PATH}")

if __name__ == "__main__":
    prepare_dashboard_csv()
