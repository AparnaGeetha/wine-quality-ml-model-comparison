# app.py
"""
Streamlit app for Wine Quality ML Model Comparison

- Loads pretrained models from model/trained_models/
- Loads preprocessing objects (imputer, scaler)
- Upload a CSV (comma or semicolon separated) or use data/wine_quality_merged.csv
- Preprocesses numeric features, runs predictions, shows metrics and plots
- Allows downloading predictions CSV

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from io import BytesIO
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality — Model Demo", layout="wide")

# Constants
MODEL_DIR = "model/trained_models"
MERGED_CSV_PATH = "data/wine_quality_merged.csv"
PREP_OBJ_NAME = "preprocessing_imputer_scaler.pkl"
TARGET_COL = "quality_label"  # name of target used in training

# ---------- Helper functions ----------
@st.cache_data
def available_models():
    if not os.path.exists(MODEL_DIR):
        return []
    files = os.listdir(MODEL_DIR)
    models = [f for f in files if f.endswith(".pkl") and f != PREP_OBJ_NAME]
    return sorted([os.path.splitext(m)[0] for m in models])

@st.cache_resource
def load_model_pickle(model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_prep():
    """
    Returns (imputer, scaler) tuple.
    """
    path = os.path.join(MODEL_DIR, PREP_OBJ_NAME)
    # 1) try loading pickled objects
    if os.path.exists(path):
        try:
            prep = joblib.load(path)
            # sanity check: should be tuple of (imputer, scaler)
            if isinstance(prep, tuple) and len(prep) == 2:
                return prep
            # if format unexpected, fall through to rebuild
        except Exception as e:
            # log the exception for debugging but continue to rebuild
            st.warning("Could not load saved preprocessing objects (pickle incompatibility). Rebuilding from dataset.")
            # Optionally show the traceback in the Streamlit logs
            print("Preprocessing load error:", e)
            traceback.print_exc()

    # 2) fallback: fit new imputer + scaler on merged CSV (if available)
    if os.path.exists(MERGED_CSV_PATH):
        try:
            df_all = read_csv_flex(MERGED_CSV_PATH)
            # drop target if present
            if TARGET_COL in df_all.columns:
                df_all = df_all.drop(columns=[TARGET_COL])
            if 'quality' in df_all.columns:
                df_all = df_all.drop(columns=['quality'])
            num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) == 0:
                raise ValueError("Merged CSV has no numeric columns to fit preprocessing.")
            imputer = SimpleImputer(strategy='median')
            scaler = StandardScaler()
            X_num = df_all[num_cols].copy()
            X_imp = imputer.fit_transform(X_num)
            scaler.fit(X_imp)
            # Save these new preprocessing objects so future loads work
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                joblib.dump((imputer, scaler), path)
                print(f"Rebuilt and saved preprocessing objects to {path}")
            except Exception as e:
                print("Warning: failed to save rebuilt preprocessing objects:", e)
            return (imputer, scaler)
        except Exception as e:
            st.error(f"Failed to rebuild preprocessing from {MERGED_CSV_PATH}: {e}")
            traceback.print_exc()
            return None
    else:
        st.error("Preprocessing objects missing and merged dataset not found. Run training script to create preprocessing objects.")
        return None
def read_csv_flex(path_or_buffer):
    """
    Try reading csv; handle comma or semicolon delimiters.
    Accepts a file path (str) or uploaded file (BytesIO / UploadedFile).
    """
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        # retry with semicolon
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, sep=';')
        except Exception as e:
            raise e

def preprocess_df_safe(df, prep_objects):
    """
    Preprocess numeric columns using saved (imputer, scaler).
    Ensures target column is excluded before transform.
    Returns processed DataFrame and list of numeric columns used.
    """
    imputer, scaler = prep_objects

    # Work on a copy
    df_local = df.copy()

    # Remove target column (if present) so it's not passed to imputer/scaler
    if TARGET_COL in df_local.columns:
        df_local = df_local.drop(columns=[TARGET_COL])

    # Also remove original 'quality' if present (the raw score)
    if 'quality' in df_local.columns:
        df_local = df_local.drop(columns=['quality'])

    # Select numeric columns only
    num_cols = df_local.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found in uploaded data after removing target columns.")

    # Check that the number of numeric cols matches scaler expectation
    # scaler.scale_ length gives number of features it was fitted on
    try:
        expected_n_features = scaler.scale_.shape[0]
    except Exception:
        expected_n_features = None

    if expected_n_features is not None and expected_n_features != len(num_cols):
        # Try a best-effort alignment by intersection in case columns are same but order different
        # This will only work if counts match or if the scaler was fitted on the same set.
        raise ValueError(
            f"Number of numeric columns in uploaded data ({len(num_cols)}) "
            f"does not match number of features expected by the preprocessing objects ({expected_n_features}). "
            "Ensure you upload the same columns used during training (physicochemical features + wine_type)."
        )

    # Now apply imputer and scaler
    X_num = df_local[num_cols].copy()
    X_imp = imputer.transform(X_num)  # may raise if column names/order mismatch significantly
    X_scaled = scaler.transform(X_imp)
    X_proc = pd.DataFrame(X_scaled, columns=num_cols, index=df_local.index)
    return X_proc, num_cols

def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return None

# ---------- App layout ----------
st.title("🍷 Wine Quality — ML Model Demo")
st.markdown("Upload a CSV (semicolon or comma separated) or use the merged dataset. Select a saved model and click **Run prediction**.")

# Sidebar controls
with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_merged = False
    if uploaded is None and os.path.exists(MERGED_CSV_PATH):
        use_merged = st.checkbox("Use merged dataset (data/wine_quality_merged.csv)", value=True)
    model_list = available_models()
    if not model_list:
        st.warning("No models found in model/trained_models/. Run training script first.")
    model_choice = st.selectbox("Choose model", options=model_list) if model_list else None
    show_raw = st.checkbox("Show raw data preview", value=False)
    run_btn = st.button("Run prediction")

# Load input DataFrame
df = None
if uploaded is not None:
    try:
        df = read_csv_flex(uploaded)
        st.sidebar.success("Uploaded file loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {e}")
elif use_merged:
    try:
        df = read_csv_flex(MERGED_CSV_PATH)
        st.sidebar.success(f"Loaded {MERGED_CSV_PATH}")
    except Exception as e:
        st.sidebar.error(f"Failed to read merged dataset: {e}")

if df is not None and show_raw:
    st.subheader("Raw data preview")
    st.dataframe(df.head())

# Run prediction flow
if run_btn:
    if df is None:
        st.error("No dataset selected. Upload a CSV or enable 'Use merged dataset'.")
    elif model_choice is None:
        st.error("No model selected.")
    else:
        prep = load_prep()
        if prep is None:
            st.error(f"Preprocessing object not found at {os.path.join(MODEL_DIR, PREP_OBJ_NAME)}. Re-run training to create it.")
        else:
            try:
                # Copy to avoid modifying original
                df_work = df.copy()

                # If original 'quality' column exists but not 'quality_label', create it
                if 'quality' in df_work.columns and TARGET_COL not in df_work.columns:
                    df_work[TARGET_COL] = (df_work['quality'] >= 7).astype(int)

                # Preprocess numeric columns safely (this excludes the target)
                X_proc, used_cols = preprocess_df_safe(df_work, prep)

                # Load model
                model = load_model_pickle(model_choice)
                if model is None:
                    st.error(f"Could not load model: {model_choice}")
                    st.stop()

                # Run predictions
                y_pred = model.predict(X_proc)
                y_proba = None
                try:
                    proba = model.predict_proba(X_proc)
                    if proba.shape[1] == 2:
                        y_proba = proba[:, 1]
                except Exception:
                    y_proba = None

                # Prepare results DataFrame: use original df but drop raw target column if present to avoid duplication
                results = df_work.reset_index(drop=True).copy()
                results['predicted_label'] = y_pred
                if y_proba is not None:
                    results['predicted_proba'] = y_proba

                # Show summary
                st.subheader("Prediction Summary")
                st.write(f"Model: **{model_choice}**")
                st.write(f"Records: {len(results)}")
                st.write(results['predicted_label'].value_counts().to_frame("count"))

                # If true labels present, compute metrics
                if TARGET_COL in results.columns:
                    y_true = results[TARGET_COL].values
                    y_pred_arr = results['predicted_label'].values

                    acc = accuracy_score(y_true, y_pred_arr)
                    prec = precision_score(y_true, y_pred_arr, zero_division=0)
                    rec = recall_score(y_true, y_pred_arr, zero_division=0)
                    f1 = f1_score(y_true, y_pred_arr, zero_division=0)
                    mcc = matthews_corrcoef(y_true, y_pred_arr)
                    auc_val = safe_auc(y_true, results['predicted_proba']) if 'predicted_proba' in results.columns else None

                    st.subheader("Evaluation Metrics (true labels present)")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"**Accuracy:** {acc:.4f}")
                        st.write(f"**Precision:** {prec:.4f}")
                        st.write(f"**Recall:** {rec:.4f}")
                    with c2:
                        st.write(f"**F1-score:** {f1:.4f}")
                        st.write(f"**MCC:** {mcc:.4f}")
                        st.write(f"**AUC:** {auc_val:.4f}" if auc_val is not None else "**AUC:** N/A")

                    st.subheader("Classification Report")
                    st.text(classification_report(y_true, y_pred_arr, zero_division=0))

                    # Confusion matrix
                    cm = confusion_matrix(y_true, y_pred_arr)
                    fig_cm, ax_cm = plt.subplots(figsize=(4,4))
                    ax_cm.imshow(cm, interpolation='nearest')
                    ax_cm.set_title("Confusion Matrix")
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    ax_cm.xaxis.set_ticklabels(['0','1'])
                    ax_cm.yaxis.set_ticklabels(['0','1'])
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax_cm.text(j, i, cm[i,j], ha="center", va="center")
                    st.pyplot(fig_cm)

                    # ROC curve if probabilities exist
                    if 'predicted_proba' in results.columns:
                        try:
                            fpr, tpr, _ = roc_curve(y_true, results['predicted_proba'])
                            roc_auc = auc(fpr, tpr)
                            fig_roc, ax_roc = plt.subplots(figsize=(5,4))
                            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
                            ax_roc.plot([0,1],[0,1], linestyle='--', color='gray')
                            ax_roc.set_title("ROC Curve")
                            ax_roc.set_xlabel("False Positive Rate")
                            ax_roc.set_ylabel("True Positive Rate")
                            ax_roc.legend(loc="lower right")
                            st.pyplot(fig_roc)
                        except Exception as e:
                            st.write("Could not compute ROC curve:", e)

                # Show sample predictions and download
                st.subheader("Sample Predictions")
                st.dataframe(results.head(20))

                csv_bytes = results.to_csv(index=False).encode('utf-8')
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error during prediction: {e}")