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
st.title("Credit Risk Analysis: Real-time & Batch Prediction with Visualizations")

# ------- Sidebar: Upload main data --------
st.sidebar.header("Step 1: Upload Training Data")
main_file = st.sidebar.file_uploader("Upload `cs-training.csv` (required for model training)", type=["csv"])

# ------- Preprocessing Function ---------
def preprocess(df):
    df['MonthlyIncome'].replace(0, np.nan, inplace=True)
    df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)
    df['NumberOfDependents'].fillna(df['NumberOfDependents'].median(), inplace=True)
    X = df.drop('SeriousDlqin2yrs', axis=1)
    y = df['SeriousDlqin2yrs']
    X['DebtToIncomeRatio'] = X['DebtRatio'] / (X['MonthlyIncome'] + 1)
    X['LatePaymentFlag'] = (X['NumberOfTime30-59DaysPastDueNotWorse'] > 0).astype(int)
    return X, y

# ------- Train Models If Data Uploaded -------
if main_file is not None:
    df = pd.read_csv(main_file, index_col=0)
    st.write("#### Preview of Training Data", df.head())
    X, y = preprocess(df)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    feature_list = X_train.columns

    # ------ TABS ------
    tab1, tab2 = st.tabs(["Single (Manual) Prediction", "Batch File Prediction"])

    # ------- TAB 1: MANUAL (REAL-TIME) ---------
    with tab1:
        st.header("Single Customer Prediction (with Real-Time Visualizations)")
        with st.form("manual_form"):
            cols = st.columns(2)
            RevolvingUtilizationOfUnsecuredLines = cols[0].slider("Revolving Utilization of Unsecured Lines", 0.0, 10.0, 0.5)
            age = cols[1].slider("Age", 18, 100, 35)
            NumberOfTime30_59DaysPastDueNotWorse = cols[0].slider("Times 30-59 Days Past Due", 0, 20, 0)
            DebtRatio = cols[1].slider("Debt Ratio", 0.0, 100.0, 5.0)
            MonthlyIncome = cols[0].slider("Monthly Income", 0.0, 50000.0, 5000.0, step=100.0)
            NumberOfOpenCreditLinesAndLoans = cols[1].slider("Open Credit Lines and Loans", 0, 30, 2)
            NumberOfTimes90DaysLate = cols[0].slider("Times 90 Days Late", 0, 20, 0)
            NumberRealEstateLoansOrLines = cols[1].slider("Real Estate Loans or Lines", 0, 10, 1)
            NumberOfTime60_89DaysPastDueNotWorse = cols[0].slider("Times 60-89 Days Past Due", 0, 20, 0)
            NumberOfDependents = cols[1].slider("Number of Dependents", 0, 10, 0)
            submit = st.form_submit_button("Predict")

        if submit:
            manual_input = pd.DataFrame([[
                RevolvingUtilizationOfUnsecuredLines, age, NumberOfTime30_59DaysPastDueNotWorse,
                DebtRatio, MonthlyIncome, NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
                NumberRealEstateLoansOrLines, NumberOfTime60_89DaysPastDueNotWorse, NumberOfDependents
            ]], columns=[
                'RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'
            ])
            manual_input['DebtToIncomeRatio'] = manual_input['DebtRatio'] / (manual_input['MonthlyIncome'] + 1)
            manual_input['LatePaymentFlag'] = (manual_input['NumberOfTime30-59DaysPastDueNotWorse'] > 0).astype(int)
            risk_rf = rf.predict_proba(manual_input)[0][1]
            risk_xgb = xgb.predict_proba(manual_input)[0][1]

            st.subheader("Results")
            st.write(f"**Random Forest Default Probability:** `{risk_rf:.2%}`")
            st.write(f"**XGBoost Default Probability:** `{risk_xgb:.2%}`")
            st.write(f"**High Risk (RF > 50%)?** {'Yes' if risk_rf > 0.5 else 'No'}")
            st.write(f"**High Risk (XGB > 50%)?** {'Yes' if risk_xgb > 0.5 else 'No'}")

        # --- Visualizations ---
        st.divider()
        st.write("### Model Visualizations (Based on Test Data)")

        col3, col4 = st.columns(2)
        with col3:
            st.write("**Confusion Matrix: Random Forest**")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, rf.predict(X_test), ax=ax)
            st.pyplot(fig)
        with col4:
            st.write("**ROC Curve: Random Forest**")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax)
            st.pyplot(fig)

        col5, col6 = st.columns(2)
        with col5:
            st.write("**Confusion Matrix: XGBoost**")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, xgb.predict(X_test), ax=ax)
            st.pyplot(fig)
        with col6:
            st.write("**ROC Curve: XGBoost**")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(xgb, X_test, y_test, ax=ax)
            st.pyplot(fig)

        # Feature Importances
        st.write("### Feature Importances")
        col7, col8 = st.columns(2)
        with col7:
            st.write("**Random Forest**")
            importances_rf = rf.feature_importances_
            indices_rf = np.argsort(importances_rf)[::-1]
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(range(10), importances_rf[indices_rf][:10])
            ax.set_xticks(range(10))
            ax.set_xticklabels(feature_list[indices_rf][:10], rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        with col8:
            st.write("**XGBoost**")
            importances_xgb = xgb.feature_importances_
            indices_xgb = np.argsort(importances_xgb)[::-1]
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.bar(range(10), importances_xgb[indices_xgb][:10])
            ax.set_xticks(range(10))
            ax.set_xticklabels(feature_list[indices_xgb][:10], rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

    # ------- TAB 2: BATCH FILE UPLOAD (PREDICTION) -----------
    with tab2:
        st.header("Batch File Prediction")
        # Sample download
        st.markdown("Need a template? [Download sample input CSV](sandbox:/sample_input.csv)")
        batch_file = st.file_uploader("Upload customer CSV for batch prediction", type=["csv"], key="batch")
        if batch_file is not None:
            test_df = pd.read_csv(batch_file)
            if 'SeriousDlqin2yrs' in test_df.columns:
                test_df = test_df.drop(columns=['SeriousDlqin2yrs'])
            test_df['DebtToIncomeRatio'] = test_df['DebtRatio'] / (test_df['MonthlyIncome'] + 1)
            test_df['LatePaymentFlag'] = (test_df['NumberOfTime30-59DaysPastDueNotWorse'] > 0).astype(int)
            X_pred = test_df[X_train.columns]
            test_df["RF_Default_Prob"] = rf.predict_proba(X_pred)[:,1]
            test_df["XGB_Default_Prob"] = xgb.predict_proba(X_pred)[:,1]
            test_df["RF_HighRisk"] = np.where(test_df["RF_Default_Prob"] > 0.5, "Yes", "No")
            test_df["XGB_HighRisk"] = np.where(test_df["XGB_Default_Prob"] > 0.5, "Yes", "No")
            st.write("#### Prediction Results (First 10 Rows)", test_df.head(10))

            # Download
            csv = test_df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", csv, "batch_predictions.csv", "text/csv")

            # --- Batch-level Visualizations (on predictions) ---
            st.write("### Batch Prediction High Risk Distribution")
            fig, ax = plt.subplots()
            test_df['RF_HighRisk'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
            plt.title("High Risk Customers (Random Forest)")
            st.pyplot(fig)

            fig, ax = plt.subplots()
            test_df['XGB_HighRisk'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
            plt.title("High Risk Customers (XGBoost)")
            st.pyplot(fig)
        # Optional: show template download button
        sample_df = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': 0.5, 'age': 35, 'NumberOfTime30-59DaysPastDueNotWorse': 0,
            'DebtRatio': 5, 'MonthlyIncome': 5000, 'NumberOfOpenCreditLinesAndLoans': 2,
            'NumberOfTimes90DaysLate': 0, 'NumberRealEstateLoansOrLines': 1,
            'NumberOfTime60-89DaysPastDueNotWorse': 0, 'NumberOfDependents': 0
        }])
        st.download_button("Download Sample Input CSV", sample_df.to_csv(index=False), "sample_input.csv")
else:
    st.info("**Please upload the main `cs-training.csv` dataset in the sidebar to use this app.**")
