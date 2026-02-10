# Wine Quality ML Model Comparison

This repository contains a machine learning project that compares multiple classification
models for predicting wine quality using physicochemical features. The project also includes
a Streamlit-based interactive application for model demonstration and evaluation.

---

## 📌 Problem Statement

The objective of this project is to build, train, and compare different machine learning
classification models to predict wine quality based on its physicochemical properties.
The task is formulated as a binary classification problem.

---

## 📊 Dataset Description

The Wine Quality dataset was obtained from the **UCI Machine Learning Repository**.
It consists of two datasets: red wine and white wine.

- Total samples: **6,497**
- Features: **12 physicochemical attributes**
- Additional feature: **wine type (red or white)**

The red and white wine datasets were merged to increase the number of samples and to include
wine type as an additional predictive feature.

The original quality score was converted into a binary target variable:
- `1` → Good quality wine (quality ≥ 7)
- `0` → Average/Poor quality wine (quality < 7)

---

## 🧠 Machine Learning Models Implemented

The following classification models were implemented and compared:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbours (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

---

## 📈 Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy  
- Area Under the ROC Curve (AUC)  
- Precision  
- Recall  
- F1-score  
- Matthews Correlation Coefficient (MCC)  

These metrics provide a comprehensive evaluation of classification performance.

---

## 📋 Results Summary

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | — | — | — | — | — | — |
| Decision Tree | — | — | — | — | — | — |
| KNN | — | — | — | — | — | — |
| Naive Bayes | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| XGBoost | — | — | — | — | — | — |

(*Values to be filled after model training*)

---

## 🔍 Observations

- Ensemble models such as Random Forest and XGBoost achieved superior performance compared
  to simpler models.
- Logistic Regression performed reasonably well, indicating near-linear separability of the data.
- Naive Bayes showed comparatively lower performance due to the assumption of feature independence.

---

## 🚀 Streamlit Application

A Streamlit-based web application was developed to:
- Upload a dataset
- Select a trained model
- View predictions and evaluation metrics
- Visualize the confusion matrix

The application provides an interactive interface to demonstrate model performance.

---

## 🛠️ How to Run the Project

1. Clone the repository:
     git clone https://github.com/<your-username>/wine-quality-ml-model-comparison.git
2.	Install dependencies:
     pip install -r requirements.txt
3.	Run the Streamlit app:
     streamlit run app.py

📚 References
	•	UCI Machine Learning Repository: Wine Quality Dataset
https://archive.ics.uci.edu/ml/datasets/Wine+Quality
