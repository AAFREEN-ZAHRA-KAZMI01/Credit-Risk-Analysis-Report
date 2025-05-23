import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Analysis", layout="wide")

st.title("Credit Risk Analysis & Prediction App")
st.write("Upload your data, train models, and predict credit risk with full visualizations.")

# Sidebar: File uploader
st.sidebar.header("1. Upload Data")
data_file = st.sidebar.file_uploader("Upload cs-training.csv", type=["csv"])

# Helper function for preprocessing
def preprocess(df):
    df['MonthlyIncome'].replace(0, np.nan, inplace=True)
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']
    X['DebtToIncomeRatio'] = X['DebtRatio'] / (X['MonthlyIncome'] + 1)
    X['LatePaymentFlag'] = (X['NumberOfTime30-59DaysPastDueNotWorse'] > 0).astype(int)
    return X, y

# Main logic
if data_file is not None:
    df = pd.read_csv(data_file, index_col=0)
    st.write("### Sample Data", df.head())
    X, y = preprocess(df)

    # Balance data
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    st.sidebar.header("2. Single Customer Prediction")
    with st.sidebar.form(key="predict_form"):
        RevolvingUtilizationOfUnsecuredLines = st.slider("Revolving Utilization of Unsecured Lines", 0.0, 10.0, 0.5)
        age = st.slider("Age", 18, 100, 35)
        NumberOfTime30_59DaysPastDueNotWorse = st.slider("Times 30-59 Days Past Due", 0, 20, 0)
        DebtRatio = st.slider("Debt Ratio", 0.0, 100.0, 5.0)
        MonthlyIncome = st.slider("Monthly Income", 0.0, 50000.0, 5000.0, step=100.0)
        NumberOfOpenCreditLinesAndLoans = st.slider("Open Credit Lines and Loans", 0, 30, 2)
        NumberOfTimes90DaysLate = st.slider("Times 90 Days Late", 0, 20, 0)
        NumberRealEstateLoansOrLines = st.slider("Real Estate Loans or Lines", 0, 10, 1)
        NumberOfTime60_89DaysPastDueNotWorse = st.slider("Times 60-89 Days Past Due", 0, 20, 0)
        NumberOfDependents = st.slider("Number of Dependents", 0, 10, 0)
        submit = st.form_submit_button("Predict Risk")

    if submit:
        input_df = pd.DataFrame([[
            RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30_59DaysPastDueNotWorse,
            DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
            NumberRealEstateLoansOrLines, NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents
        ]], columns=[
            'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
            'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
        ])
        input_df['DebtToIncomeRatio'] = input_df['DebtRatio'] / (input_df['MonthlyIncome'] + 1)
        input_df['LatePaymentFlag'] = (input_df['NumberOfTime30-59DaysPastDueNotWorse'] > 0).astype(int)
        risk_prob_rf = rf.predict_proba(input_df)[0][1]
        risk_prob_xgb = xgb.predict_proba(input_df)[0][1]
        st.sidebar.markdown(f"**Random Forest Default Probability:** `{risk_prob_rf:.2%}`")
        st.sidebar.markdown(f"**XGBoost Default Probability:** `{risk_prob_xgb:.2%}`")
        st.sidebar.markdown(f"**High Risk (RF > 50%)?** {'Yes' if risk_prob_rf > 0.5 else 'No'}")
        st.sidebar.markdown(f"**High Risk (XGB > 50%)?** {'Yes' if risk_prob_xgb > 0.5 else 'No'}")

    st.header("3. Model Results and Visualizations")

    # 1. Confusion Matrix
    y_pred_rf = rf.predict(X_test)
    y_pred_xgb = xgb.predict(X_test)

    st.subheader("Confusion Matrix: Random Forest")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, ax=ax)
    st.pyplot(fig)

    st.subheader("Confusion Matrix: XGBoost")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_xgb, ax=ax)
    st.pyplot(fig)

    # 2. ROC Curve
    st.subheader("ROC Curve: Random Forest")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve: XGBoost")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(xgb, X_test, y_test, ax=ax)
    st.pyplot(fig)

    # 3. Feature Importance
    st.subheader("Feature Importance: Random Forest")
    importances = rf.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(10), importances[indices][:10])
    ax.set_xticks(range(10))
    ax.set_xticklabels(features[indices][:10], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Feature Importance: XGBoost")
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(10), importances[indices][:10])
    ax.set_xticks(range(10))
    ax.set_xticklabels(features[indices][:10], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # 4. Batch Prediction (optional for user-uploaded file)
    st.header("4. Batch Prediction")
    st.write("Upload a file (CSV) of customers to get batch risk predictions:")

    pred_file = st.file_uploader("Upload customers file for batch scoring", type=["csv"], key="batch")
    if pred_file is not None:
        test_df = pd.read_csv(pred_file)
        test_df['DebtToIncomeRatio'] = test_df['DebtRatio'] / (test_df['MonthlyIncome'] + 1)
        test_df['LatePaymentFlag'] = (test_df['NumberOfTime30-59DaysPastDueNotWorse'] > 0).astype(int)
        # Must match the features used during training
        X_pred = test_df[X_train.columns]
        test_df["RF_Default_Prob"] = rf.predict_proba(X_pred)[:,1]
        test_df["XGB_Default_Prob"] = xgb.predict_proba(X_pred)[:,1]
        test_df["RF_HighRisk"] = np.where(test_df["RF_Default_Prob"] > 0.5, "Yes", "No")
        test_df["XGB_HighRisk"] = np.where(test_df["XGB_Default_Prob"] > 0.5, "Yes", "No")
        st.write(test_df.head())
        csv = test_df.to_csv(index=False).encode()
        st.download_button("Download Results CSV", csv, "predictions.csv", "text/csv")
else:
    st.info("Please upload the cs-training.csv dataset first (sidebar) to get started.")
