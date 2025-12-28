# Customer Churn Prediction System

This project implements an end-to-end machine learning solution to predict customer churn and support data-driven retention strategies. It combines data preprocessing, model training, explainability, and a deployed web interface built using Streamlit.

---

## Overview

Customer churn is a critical problem for subscription-based businesses, as acquiring new customers is often more expensive than retaining existing ones.  
The goal of this project is to **identify customers who are likely to churn** and provide **clear, actionable insights** that can help businesses take proactive measures.

Rather than focusing only on model accuracy, this project emphasizes:
- Explainability
- Business relevance
- Deployment readiness

---

## Key Features

- Predicts churn probability for individual customers
- Categorizes churn risk into Low, Medium, and High
- Displays top risk drivers influencing the prediction
- Provides business-oriented retention recommendations
- Clean and modern Streamlit-based user interface
- Optional SHAP-based explainability analysis (offline)
- Deployment-ready project structure

---

## Technology Stack

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Joblib  
- **Machine Learning Model:** Logistic Regression  
- **Web Framework:** Streamlit  
- **Explainability:** Model coefficients and SHAP (offline analysis)  
- **Version Control:** GitHub  

---

## Project Structure

Customer-Churn-Prediction/
│
├── app/
│ └── app.py # Streamlit application
│
├── models/
│ ├── churn_prediction_model.pkl
│ ├── feature_columns.pkl
│ └── scaler.pkl
│
├── data/
│ ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│ ├── cleaned_telco_churn.csv
│ ├── X_features.csv
│ └── y_target.csv
│
├── notebooks/
│ ├── 01_dataset_understanding.ipynb
│ ├── 02_data_cleaning_preprocessing.ipynb
│ ├── 03_exploratory_data_analysis.ipynb
│ ├── 04_feature_engineering.ipynb
│ ├── 05_model_building.ipynb
│ ├── 06_model_evaluation.ipynb
│ ├── 07_model_explainability.ipynb
│ ├── 07b_optional_shap_explainability.ipynb
│ ├── 08_retention_strategy.ipynb
│ └── 09_model_saving.ipynb
│
├── requirements.txt
└── README.md
