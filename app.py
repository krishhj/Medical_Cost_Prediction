# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and columns
model = joblib.load("insurance_model.pkl")
columns = json.load(open("columns.json"))

st.set_page_config(page_title="Medical Cost Predictor", layout="centered")
st.title("Medical Insurance Cost Predictor")

# Sidebar inputs
st.sidebar.header("Enter patient details")
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.slider("BMI", 10.0, 60.0, 25.0)
children = st.sidebar.slider("Children", 0, 10, 0)
sex = st.sidebar.selectbox("Sex", ["male","female"])
smoker = st.sidebar.selectbox("Smoker", ["yes","no"])
region = st.sidebar.selectbox("Region", ["northeast","northwest","southeast","southwest"])

# Prepare input dataframe with same columns/order as training
data = dict.fromkeys(columns, 0)
data["age"] = age
data["bmi"] = bmi
data["children"] = children
data["sex_male"] = 1 if sex == "male" else 0
data["smoker_yes"] = 1 if smoker == "yes" else 0
# region dummies: northeast is the base (all zero)
for r in ["region_northwest","region_southeast","region_southwest"]:
    data[r] = 1 if r.split("_")[1] in region else 0  # simple mapping

input_df = pd.DataFrame([data], columns=columns)

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    pred = max(pred, 0) 
    st.success(f"Estimated annual medical charges: ${pred:,.2f}")

    # show simple comparison
    st.write("Feature contributions (coefficients * input values):")
    coefs = pd.Series(model.coef_, index=columns)
    contrib = coefs * input_df.iloc[0]
    contrib = contrib.sort_values(ascending=False)
    st.table(contrib.head(8).to_frame(name="contribution"))

# Optional visualizations
if st.checkbox("Show dataset charge distribution (for context)"):
    df = pd.read_csv("insurance.csv")
    fig, ax = plt.subplots()
    sns.histplot(df["charges"], kde=True, ax=ax)
    st.pyplot(fig)
