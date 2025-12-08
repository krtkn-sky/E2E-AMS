from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ----- Paths -----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "purchase_model.pkl")

# ----- Load model -----
model = joblib.load(MODEL_PATH)

# ----- FastAPI app -----
app = FastAPI(
    title="E2E-AMS Purchase Prediction API",
    description="Predicts if a user will purchase based on event data",
    version="1.0"
)

# ----- Pydantic input model -----
class EventData(BaseModel):
    hour: int
    day_of_week: int
    is_weekend: int
    interaction_score: float
    day: int
    weekday: int
    evt_add_to_cart: int
    evt_click: int
    evt_view: int
    user_total_score: float
    user_avg_score: float
    unique_products_viewed: int
    product_view_count: int

# ----- Prediction endpoint -----
@app.post("/predict")
def predict_purchase(event: EventData):
    # Convert input to DataFrame
    df = pd.DataFrame([event.dict()])
    
    # Predict probability
    prob = model.predict_proba(df)[:,1][0]
    
    return {"purchase_probability": prob}
