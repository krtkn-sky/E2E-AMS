import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from time import sleep

# ---------- CONFIG ----------
API_URL = "http://127.0.0.1:8000/predict"
st.set_page_config(page_title="E2E-AMS Dashboard", layout="wide")

st.title("E2E-AMS: User Purchase Prediction Dashboard")
st.write("Upload CSV with user events (use `data/events_features_dashboard.csv` for best results)")

# -------------------------------
# 1. Upload event data
uploaded_file = st.file_uploader("Upload CSV with user events", type="csv")

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

def prepare_payload_from_row(row):
    # Convert booleans to ints and ensure fields exist
    payload = {
        "hour": safe_int(row.get("hour", 0)),
        "day_of_week": safe_int(row.get("day_of_week", 0)),
        "is_weekend": safe_int(row.get("is_weekend", 0)),
        "interaction_score": safe_float(row.get("interaction_score", 0.0)),
        "day": safe_int(row.get("day", 0)),
        "weekday": safe_int(row.get("weekday", row.get("day_of_week", 0))),
        "evt_add_to_cart": safe_int(row.get("evt_add_to_cart", 0)),
        "evt_click": safe_int(row.get("evt_click", 0)),
        "evt_view": safe_int(row.get("evt_view", 0)),
        "user_total_score": safe_float(row.get("user_total_score", 0.0)),
        "user_avg_score": safe_float(row.get("user_avg_score", 0.0)),
        "unique_products_viewed": safe_int(row.get("unique_products_viewed", 0)),
        "product_view_count": safe_int(row.get("product_view_count", 0))
    }
    return payload

if uploaded_file:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    st.subheader("Event Data Preview")
    st.dataframe(df.head())

    # Show column info so user can confirm
    st.write("Columns in uploaded file:", df.columns.tolist())

    # -------------------------------
    # 2. Make predictions via API (robust)
    st.subheader("Predicted Purchase Probabilities")

    # Prepare records
    records = df.to_dict(orient="records")
    preds = []
    progress = st.progress(0)
    status_text = st.empty()

    for i, rec in enumerate(records):
        payload = prepare_payload_from_row(rec)
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # handle different key names the API might return
                prob = None
                if isinstance(data, dict):
                    prob = data.get("purchase_probability") or data.get("probability") or data.get("probabilities") or data.get("purchase_prob")
                # fallback if API changed format
                if prob is None:
                    # try top-level numeric
                    try:
                        prob = float(data)
                    except:
                        prob = None
                preds.append(prob)
            else:
                preds.append(None)
        except Exception as e:
            preds.append(None)

        # update progress
        progress.update((i + 1) / len(records))
        status_text.text(f"Processed {i + 1}/{len(records)} rows")
        # small sleep to keep UI responsive for large files (tiny)
        # sleep(0.001)

    # attach predictions
    df["purchase_probability"] = preds

    # Show results (first 20)
    st.subheader("Prediction Table (sample)")
    st.dataframe(df[["user_id", "product_id", "purchase_probability"]].head(20))

    # -------------------------------
    # 3. Visualizations (same as original, but robust)
    st.subheader("Top 10 Users by Purchase Probability")
    if "purchase_probability" in df.columns:
        top_users = df.groupby("user_id")["purchase_probability"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(top_users)
    else:
        st.info("No purchase_probability column found to compute top users.")

    st.subheader("Top 10 Products by Purchase Probability")
    if "purchase_probability" in df.columns:
        top_products = df.groupby("product_id")["purchase_probability"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(top_products)
    else:
        st.info("No purchase_probability column found to compute top products.")

    st.subheader("Probability Distribution")
    if "purchase_probability" in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df["purchase_probability"].dropna(), bins=20, kde=True)
        plt.title("Predicted Purchase Probability Distribution")
        plt.xlabel("Purchase probability")
        plt.ylabel("Count")
        st.pyplot(plt)
    else:
        st.info("No purchase_probability column found for distribution plot.")

    # -------------------------------
    # 4. Download predicted CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="predicted_output.csv",
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file (events_features_dashboard.csv recommended) to begin.")
