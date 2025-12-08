import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ----- PATHS -----
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
FEATURE_PATH = os.path.join(BASE_DIR, "data", "events_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "purchase_model.pkl")

# Load feature-engineered dataset
df = pd.read_csv(FEATURE_PATH)
print("Loaded feature dataset:", df.shape)

# ----- 1. Define target & features -----
TARGET = "label_purchase"

# Drop identifiers and leakage features
leak_features = ["timestamp", "user_id", "product_id", "evt_purchase"]  # remove evt_purchase to avoid leakage
X = df.drop(columns=leak_features + [TARGET])
y = df[TARGET]

print("Features shape:", X.shape)
print("Target distribution:\n", y.value_counts())

# ----- 2. Train-test split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----- 3. Train RandomForest with class balancing -----
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced',  # handle class imbalance
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model trained.")

# ----- 4. Evaluate -----
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----- 5. Feature Importance -----
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 important features:\n", importances.head(10))

# ----- 6. Save model -----
joblib.dump(model, MODEL_PATH)
print("Model saved at:", MODEL_PATH)
