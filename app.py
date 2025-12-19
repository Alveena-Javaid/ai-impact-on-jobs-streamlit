import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load saved artifacts
# --------------------------
MODEL_PATH = "./artifacts/rf_pipeline.joblib"
FEATURES_PATH = "./artifacts/features_list.pkl"
CLASSES_PATH = "./artifacts/classes.pkl"

pipeline = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
classes = joblib.load(CLASSES_PATH)

# --------------------------
# Streamlit App
# --------------------------
st.title("📊 AI Impact on Jobs 2030 – Prediction App")
st.write("This Streamlit application demonstrates EDA, model predictions, "
         "and analysis based on the **AI_Impact_on_Jobs_2030** dataset.")

# -----------------------------------
# Sidebar Navigation
# -----------------------------------
menu = st.sidebar.radio("Navigation", ["Introduction", "EDA", "Model Prediction", "Conclusion"])

# -----------------------------------
# Introduction
# -----------------------------------
if menu == "Introduction":
    st.header("📘 Project Introduction")
    st.write("""
    This project analyzes how AI could impact different jobs by 2030 based on:
    - Salary  
    - Experience  
    - Automation probability  
    - Tech growth  
    - Skills and education  
    """)

    st.write("A Random Forest model was trained to predict the **Risk Category** of a job.")

# -----------------------------------
# EDA SECTION
# -----------------------------------
elif menu == "EDA":
    st.header("🔍 Exploratory Data Analysis")

    uploaded_csv = st.file_uploader("Upload dataset (same CSV used for training)", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # --- Summary statistics ---
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        # --- Missing values ---
        st.subheader("Missing Values")
        st.write(df.isna().sum())

        # --- Numeric distribution ---
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.subheader("Histogram – Numeric Features")

        for col in num_cols[:5]:
            fig, ax = plt.subplots()
            ax.hist(df[col], bins=30)
            ax.set_title(f"Histogram of {col}")
            st.pyplot(fig)

        # --- Correlation heatmap ---
        if len(num_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
            st.pyplot(fig)

# -----------------------------------
# MODEL PREDICTION
# -----------------------------------
elif menu == "Model Prediction":
    st.header("🤖 Predict Job Risk Category")

    st.write("Enter the values below to predict the job's AI risk category:")

    user_input = {}

    for feat in features:
        if "Skill" in feat:
            user_input[feat] = st.slider(feat, 0, 10, 5)
        elif feat == "Education_Level":
            user_input[feat] = st.selectbox("Education Level",
                                            ["High School", "Bachelor", "Master", "PhD"])
        else:
            # numeric input
            user_input[feat] = st.number_input(feat, value=0.0)

    if st.button("Predict"):
     input_df = pd.DataFrame([user_input])

    # 🔧 Ensure column order & presence matches training
    input_df = input_df.reindex(columns=features)

    # 🔧 Fix data types
    for col in input_df.columns:
        if col == "Education_Level":
            input_df[col] = input_df[col].astype(str)
        else:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    # 🔧 Final safety: replace any remaining NaN
    input_df = input_df.fillna(0)

    prediction = pipeline.predict(input_df)[0]
    st.success(f"Predicted Risk Category: **{prediction}**")


# -----------------------------------
# Conclusion
# -----------------------------------
elif menu == "Conclusion":
    st.header("📌 Conclusion")
    st.write("""
    - Completed EDA with 15+ analyses  
    - Preprocessed dataset using scaling and one-hot encoding  
    - Trained RandomForest model  
    - Built an interactive Streamlit app  
    - App allows real-time predictions  
    """)

    st.write("🎯 Project Completed Successfully!")
