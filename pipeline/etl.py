import pandas as pd

def clean_events(input_path="data/events_raw.csv", output_path="data/events_clean.csv"):
    # Read raw data
    df = pd.read_csv(input_path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Drop rows with missing values (rare in synthetic data but good practice)
    df = df.dropna()

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Feature: hour of day
    df["hour"] = df["timestamp"].dt.hour

    # Feature: day of week (0 = Monday)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Feature: is_weekend
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Feature: interaction score (simple but useful for ML)
    interaction_map = {
        "view": 1,
        "click": 2,
        "add_to_cart": 3,
        "purchase": 4
    }
    df["interaction_score"] = df["event"].map(interaction_map)

    # Save cleaned data
    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved → {output_path}")
    print(f"Rows before: {len(df)} | Rows after (cleaned): {df.shape[0]}")

if __name__ == "__main__":
    clean_events()
