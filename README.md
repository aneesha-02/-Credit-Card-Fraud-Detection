# Credit Card Fraud Detection Using Machine Learning

## Project Overview

Credit Card Fraud Detection is a Machine Learning project designed to identify potentially fraudulent credit card transactions.

This project uses Machine Learning algorithms to classify transactions as legitimate or fraudulent. It also includes an interactive Streamlit dashboard for data visualization, model analysis, and fraud prediction.

## Project Objectives

- Analyze credit card transaction data.
- Identify fraudulent transactions.
- Handle imbalanced datasets.
- Train and evaluate Machine Learning models.
- Predict fraud using transaction details.
- Display results through an interactive dashboard.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Models

The following algorithms were used:

- Logistic Regression
- Random Forest Classifier

The Random Forest model was selected for the final fraud prediction system.

## Model Performance

The final Random Forest model achieved the following results on the test dataset using a prediction threshold of 0.3:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 93.18% |
| Recall | 83.67% |
| F1-Score | 88.17% |

These metrics help evaluate how effectively the model identifies fraudulent transactions.

## Dashboard Features

The project includes an interactive Streamlit dashboard with the following sections:

### 1. Dashboard
- Total transaction overview
- Fraud and legitimate transaction distribution
- Transaction amount visualizations

### 2. Model Analytics
- Model performance metrics
- Model comparison
- Confusion matrix
- Threshold analysis

### 3. Fraud Prediction
- Upload transaction data using a CSV file
- Predict fraudulent transactions
- View prediction results
- Download prediction results

### 4. Dataset
- Dataset preview
- Column selection
- Dataset download option

## Dataset Information

The project uses the Credit Card Fraud Detection dataset containing 284,807 transactions and 31 columns.

The dataset contains:
- Time
- Amount
- V1 to V28
- Class

The Class column indicates whether a transaction is legitimate or fraudulent.

**Note:** The original dataset is not included in this GitHub repository because its file size exceeds GitHub's 100 MB limit.

Download the dataset and place it inside the `data` folder with the filename:

`creditcard.csv`

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── fraud_detection.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── fraud_detection_model.pkl
│
└── data/
    └── creditcard.csv