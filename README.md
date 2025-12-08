# 📦 E2E-AMS --- End-to-End Automated ML System

E2E-AMS is a production-style machine learning system that covers the
complete lifecycle of an ML project --- from raw data ingestion to a
working frontend application.\
This project was built to simulate real-world ML workflows: pipelines,
automation, API deployment, and UI integration.

------------------------------------------------------------------------

## 🚀 Features

-   **Automated ETL Pipeline** for data cleaning & preprocessing\
-   **ML Model Training** and versioned model saving\
-   **FastAPI Backend** exposing a `/predict` endpoint\
-   **Streamlit Frontend** for easy user interaction\
-   **Cron Automation** for scheduled ETL + retraining\
-   **Industry-style project structure** with modular components

------------------------------------------------------------------------

## 🗂️ Project Structure

    E2E-AMS/
    ├── data/               # Raw and processed datasets
    ├── pipeline/           # ETL scripts (cleaning, transformations)
    ├── models/             # Model training scripts & saved models
    ├── api/                # FastAPI backend (prediction endpoint)
    └── frontend/           # Streamlit UI for interacting with the API

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   Python\
-   Pandas, NumPy\
-   scikit-learn\
-   FastAPI\
-   Streamlit\
-   Uvicorn\
-   Joblib / Pickle\
-   Cron (Linux automation)

------------------------------------------------------------------------

## 🔄 System Workflow

    Raw Data 
       → ETL Pipeline 
       → Processed Data 
       → Model Training 
       → Saved Model 
       → FastAPI Endpoint 
       → Streamlit Frontend

------------------------------------------------------------------------

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

``` bash
git clone <repo-link>
cd E2E-AMS
```

### 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3️⃣ Run the ETL Pipeline

``` bash
python pipeline/etl.py
```

### 4️⃣ Train the Model

``` bash
python models/train_model.py
```

### 5️⃣ Start the FastAPI Server

``` bash
uvicorn api.main:app --reload
```

### 6️⃣ Launch the Streamlit Frontend

``` bash
streamlit run frontend/app.py
```

------------------------------------------------------------------------

## ✨ Why This Project Exists

Most ML projects stop at notebooks.\
E2E-AMS was created to practice:

-   Structuring a real ML system end-to-end\
-   Building automated pipelines\
-   Deploying a model with an API\
-   Designing a simple user interface\
-   Understanding how production ML works

------------------------------------------------------------------------

## 📘 Learnings

-   How ETL pipelines are structured\
-   How APIs wrap around ML models\
-   How frontends communicate with backends\
-   How cron automation works\
-   How to build modular, reusable codebases

------------------------------------------------------------------------

## 🛤️ Future Improvements

-   Add Docker support\
-   Add proper logging & monitoring\
-   Add a database (PostgreSQL/SQLite)\
-   Add CI/CD pipeline\
-   Deploy API + frontend to cloud (AWS/GCP/Render)

------------------------------------------------------------------------

## 📄 License

This project is released under the MIT License.
