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
|------|----------:|-----:|----------:|------:|---------:|----:|
| Logistic Regression | 0.8223 | 0.8048 | 0.6147 | 0.2617 | 0.3671 | 0.3178 |
| Decision Tree | 0.8538 | 0.7749 | 0.6250 | 0.6445 | 0.6346 | 0.5434 |
| KNN | 0.8323 | 0.8264 | 0.5922 | 0.4766 | 0.5281 | 0.4314 |
| GaussianNB | 0.7346 | 0.7486 | 0.3901 | 0.6172 | 0.4781 | 0.3268 |
| Random Forest | 0.8877 | 0.9162 | 0.8090 | 0.5625 | 0.6636 | 0.6130 |
| XGBoost | 0.8869 | 0.9087 | 0.7633 | 0.6172 | 0.6825 | 0.6198 |

---

## 🔍 Observations

- Ensemble models (Random Forest and XGBoost) achieved the best overall performance.
- XGBoost obtained the highest MCC (0.6198) and F1-score (0.6825), indicating better balance between precision and recall.
- Random Forest achieved the highest AUC (0.9162), demonstrating strong classification capability.
- Logistic Regression showed high precision but low recall, suggesting conservative prediction of high-quality wines.
- Gaussian Naive Bayes performed the weakest, likely due to the assumption of feature independence.
  
---

## 🎯 Primary Evaluation Metric

Matthews Correlation Coefficient (MCC) was selected as the primary metric because the dataset is slightly imbalanced. MCC provides a balanced evaluation by considering true and false positives and negatives, making it more reliable than accuracy alone.

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
     git clone [https://github.com/AparnaGeetha/wine-quality-ml-model-comparison.git](https://github.com/AparnaGeetha/wine-quality-ml-model-comparison)
2.	Install dependencies:
     pip install -r requirements.txt
3.	Run the Streamlit app:
     streamlit run app.py

📚 References
	•	UCI Machine Learning Repository: Wine Quality Dataset
https://archive.ics.uci.edu/ml/datasets/Wine+Quality
