"""
train_models.py

Usage:
    python model/train_models.py
This will:
 - download winequality CSVs if needed
 - merge them, create binary target quality_label (quality >= 7)
 - train 6 models: LogisticRegression, DecisionTree, KNN, GaussianNB, RandomForest, XGBoost
 - save models to model/trained_models/
 - save metrics CSV to model/metrics_comparison.csv
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
import joblib
import argparse
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# Constants / file paths
# --------------------------
RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
DATA_DIR = "data"
MODEL_DIR = "model/trained_models"
MERGED_CSV = os.path.join(DATA_DIR, "wine_quality_merged.csv")
METRICS_CSV = os.path.join("model", "metrics_comparison.csv")
RANDOM_STATE = 42

# --------------------------
# Helpers
# --------------------------
def download_if_missing(url, out_path):
    if not os.path.exists(out_path):
        print(f"Downloading {url} -> {out_path} ...")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        urllib.request.urlretrieve(url, out_path)
        print("Downloaded.")
    else:
        print(f"File already exists: {out_path}")

def download_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)
    red_path = os.path.join(DATA_DIR, "winequality-red.csv")
    white_path = os.path.join(DATA_DIR, "winequality-white.csv")
    download_if_missing(RED_URL, red_path)
    download_if_missing(WHITE_URL, white_path)
    return red_path, white_path

def merge_and_prepare(red_path, white_path, save_csv=True):
    red = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')
    red['wine_type'] = 0
    white['wine_type'] = 1
    df = pd.concat([red, white], ignore_index=True)
    # binary target
    df['quality_label'] = (df['quality'] >= 7).astype(int)
    df.drop(columns=['quality'], inplace=True)
    if save_csv:
        df.to_csv(MERGED_CSV, index=False)
        print(f"Merged dataset saved to {MERGED_CSV} (shape: {df.shape})")
    return df

def preprocess(df, target_col='quality_label', test_size=0.2):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    # Separate numeric columns (all in this dataset are numeric)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # train-test split (stratify because class imbalance possible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    # Impute median, then scale
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_num = imputer.fit_transform(X_train[num_cols])
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_num = imputer.transform(X_test[num_cols])
    X_test_scaled = scaler.transform(X_test_num)
    # Wrap back into DataFrame to keep columns (optional)
    X_train_prep = pd.DataFrame(X_train_scaled, columns=num_cols, index=X_train.index)
    X_test_prep = pd.DataFrame(X_test_scaled, columns=num_cols, index=X_test.index)
    return X_train_prep, X_test_prep, y_train, y_test, (imputer, scaler), num_cols

def get_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    }
    return models

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # probabilities for AUC if available
    try:
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:,1])
        else:
            # unlikely given binary target
            auc = roc_auc_score(y_test, y_proba[:,1])
    except:
        auc = None
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": auc,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics

# --------------------------
# Main
# --------------------------
def main():
    # 1) Download and merge
    red_path, white_path = download_datasets()
    df = merge_and_prepare(red_path, white_path, save_csv=True)

    # 2) Preprocess
    X_train, X_test, y_train, y_test, prep_tools, feature_cols = preprocess(df, target_col='quality_label', test_size=0.2)
    print("Preprocessing done. Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # Ensure model dir
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save preprocessing objects for later use in app
    joblib.dump(prep_tools, os.path.join(MODEL_DIR, "preprocessing_imputer_scaler.pkl"))
    print("Saved preprocessing objects.")

    # 3) Train models
    models = get_models()
    results = []
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        # Save model pickle
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
        print(f"Saved {name}.pkl")
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        print(f"{name} metrics: Accuracy={metrics['accuracy']:.4f} AUC={metrics['auc'] if metrics['auc'] is not None else 'N/A'} F1={metrics['f1']:.4f} MCC={metrics['mcc']:.4f}")
        row = {
            "model": name,
            "accuracy": metrics['accuracy'],
            "auc": metrics['auc'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "mcc": metrics['mcc'],
            "confusion_matrix": metrics['confusion_matrix'].tolist()
        }
        results.append(row)

    # 4) Save metrics CSV
    metrics_df = pd.DataFrame(results)
    # keep confusion_matrix as text or drop for CSV summary
    metrics_df_to_save = metrics_df.drop(columns=["confusion_matrix"])
    os.makedirs(os.path.dirname(METRICS_CSV), exist_ok=True)
    metrics_df_to_save.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved metrics summary to {METRICS_CSV}")
    print("\nFull results:")
    print(metrics_df_to_save)

if __name__ == "__main__":
    main()