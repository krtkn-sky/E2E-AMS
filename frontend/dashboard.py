import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://127.0.0.1:8000/predict"

st.title("E2E-AMS: User Purchase Prediction Dashboard")

# -------------------------------
# 1. Upload event data
uploaded_file = st.file_uploader("Upload CSV with user events", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Event Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # 2. Make predictions via API
    st.subheader("Predicted Purchase Probabilities")

    def get_prediction(row):
        payload = {
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
            "is_weekend": int(row["is_weekend"]),
            "interaction_score": float(row["interaction_score"]),
            "day": int(row["day"]),
            "weekday": int(row["weekday"]),
            "evt_add_to_cart": int(row["evt_add_to_cart"]),
            "evt_click": int(row["evt_click"]),
            "evt_view": int(row["evt_view"]),
            "user_total_score": float(row["user_total_score"]),
            "user_avg_score": float(row["user_avg_score"]),
            "unique_products_viewed": int(row["unique_products_viewed"]),
            "product_view_count": int(row["product_view_count"])
        }
        response = requests.post(API_URL, json=payload)
        return response.json()["purchase_probability"]

    df["purchase_probability"] = df.apply(get_prediction, axis=1)
    st.dataframe(df[["user_id", "product_id", "purchase_probability"]].head(20))

    # -------------------------------
    # 3. Visualizations

    st.subheader("Top 10 Users by Purchase Probability")
    top_users = df.groupby("user_id")["purchase_probability"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_users)

    st.subheader("Top 10 Products by Purchase Probability")
    top_products = df.groupby("product_id")["purchase_probability"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

    st.subheader("Probability Distribution")
    plt.figure(figsize=(8,4))
    sns.histplot(df["purchase_probability"], bins=20, kde=True)
    st.pyplot(plt)
