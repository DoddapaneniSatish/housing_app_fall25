# Telco Customer Churn Prediction System

## 📌 Project Overview

This project is an end-to-end Machine Learning system for predicting customer churn using the **Telco Customer Churn Dataset**.

The project includes:

- Data preprocessing
- SQLite normalized database (3NF)
- SQL joins with Pandas
- Exploratory Data Analysis (EDA)
- Multiple ML experiments
- PCA dimensionality reduction
- Hyperparameter tuning
- MLflow + DagsHub experiment tracking
- FastAPI model serving
- Streamlit frontend
- Docker containerization
- Cloud deployment using Render

---

# 📂 Project Structure

```bash
housing_app_fall25/
│
├── api/                     # FastAPI backend
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── housing_pipeline.py
│
├── streamlit/               # Streamlit frontend
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── data/
│   ├── raw/                 # Raw dataset
│   ├── processed/
│   ├── db/
│   └── data_schema.json
│
├── models/                  # Saved ML models
│
├── metrics/                 # Experiment results
│
├── notebooks/               # Jupyter notebooks
│
├── src/                     # ML pipeline source code
│
├── docker-compose.yml
└── README.md
```

---

# 📊 Dataset

Dataset: **Telco Customer Churn**

The dataset contains customer demographic information, service usage, billing information, and churn labels.

### Target Variable

- `Churn`
  - Yes = customer left
  - No = customer stayed

---

# 🗄 Database Design (3NF)

A normalized SQLite database was created using:

- Dimension tables
- Fact table
- Foreign key relationships

### Purpose

- Reduce redundancy
- Improve data integrity
- Enable SQL joins for analytics

---

# 🔍 Exploratory Data Analysis

Performed:
- Correlation analysis
- Missing value analysis
- Feature distributions
- Class imbalance analysis
- ydata-profiling report

### Observations

- Dataset was slightly imbalanced
- Some categorical features had dominant classes
- Numerical features required scaling

---

# ⚙️ Machine Learning Experiments

## Models Used

- Logistic Regression
- Ridge Classifier
- Random Forest
- XGBoost
- LightGBM

---

# 🧠 PCA (Principal Component Analysis)

PCA was used for dimensionality reduction.

### Purpose

- Reduce feature dimensionality
- Remove redundancy
- Improve generalization

Experiments were performed:
- With PCA
- Without PCA

---

# 🔧 Hyperparameter Tuning

Hyperparameter tuning was performed using:
- Grid Search / Optuna

### Metrics Logged

- F1-score
- Precision
- Recall
- Confusion Matrix

---

# 📈 Experiment Tracking

MLflow + DagsHub were used for experiment tracking.

Tracked:
- Parameters
- Metrics
- Model artifacts
- Best models

---

# 🏆 Best Model

### Best Performing Model

- **LightGBM**
- PCA Disabled
- Hyperparameter Tuned

### Metric

Highest F1-score among all experiments.

Saved as:

```bash
models/global_best_model.pkl
```

---

# 🚀 FastAPI Backend

The FastAPI backend serves predictions using the trained model.

## Endpoints

### Health Check

```http
GET /health
```

### Prediction Endpoint

```http
POST /predict
```

---

# 🎨 Streamlit Frontend

The Streamlit application provides:
- User-friendly UI
- Real-time predictions
- API integration

---

# 🐳 Docker Containerization

Both services were containerized using Docker.

### Services

- FastAPI API
- Streamlit UI

Managed using:

```bash
docker-compose up --build
```

---

# ☁️ Cloud Deployment

Deployed on:
- Render

### Deployed Services

- FastAPI backend
- Streamlit frontend

---

# 🛠 Technologies Used

## Programming
- Python

## ML Libraries
- scikit-learn
- XGBoost
- LightGBM
- pandas
- numpy

## Backend
- FastAPI

## Frontend
- Streamlit

## Database
- SQLite

## DevOps
- Docker
- Docker Compose
- Render

## Experiment Tracking
- MLflow
- DagsHub

---

# ▶️ Running Locally

## 1. Clone Repository

```bash
git clone https://github.com/santhosh-madha/housing_app_fall25.git
cd housing_app_fall25
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux/Mac

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start Docker Services

```bash
docker-compose up --build
```

---

# 🌐 Access Applications

## FastAPI

```text
http://localhost:8000/docs
```

## Streamlit

```text
http://localhost:8501
```

---

# 📌 Example Prediction Request

```json
{
  "instances": [
    {
      "gender": "Male",
      "senior_citizen": "No",
      "partner": "Yes",
      "dependents": "No",
      "tenure": 12,
      "phone_service": "Yes",
      "multiple_lines": "No",
      "internet_service": "Fiber optic",
      "online_security": "No",
      "online_backup": "Yes",
      "device_protection": "No",
      "tech_support": "No",
      "streaming_tv": "Yes",
      "streaming_movies": "No",
      "contract_type": "Month-to-month",
      "paperless_billing": "Yes",
      "payment_method": "Electronic check",
      "monthly_charges": 70.35,
      "total_charges": 845.5
    }
  ]
}
```

---

# 📚 Key Learnings

- End-to-end ML pipeline development
- Database normalization
- Experiment tracking
- Model deployment
- Docker containerization
- Cloud deployment
- API development

---
