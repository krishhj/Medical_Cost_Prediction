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

# The training data covers age 18-64, BMI 15.96-53.13, and 0-5 children
if age > 64 or not 15.96 <= bmi <= 53.13 or children > 5:
    st.sidebar.warning("These details are outside the training data, so the estimate is less reliable.")

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
# Smoking costs far more for people who are also obese (BMI 30+)
data["obese"] = 1 if bmi >= 30 else 0
data["smoker_obese"] = data["smoker_yes"] * data["obese"]

input_df = pd.DataFrame([data], columns=columns)

labels = {
    "age": "Age", "bmi": "BMI", "children": "Children", "sex_male": "Male",
    "smoker_yes": "Smoker", "region_northwest": "Region: northwest",
    "region_southeast": "Region: southeast", "region_southwest": "Region: southwest",
    "obese": "Obese (BMI 30+)", "smoker_obese": "Smoker and obese",
}

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    pred = max(pred, 0) 
    st.success(f"Estimated annual medical charges: ${pred:,.2f}")

    # show simple comparison
    st.write("How much each factor adds to the estimate (coefficient × your value):")
    coefs = pd.Series(model.coef_, index=columns)
    contrib = (coefs * input_df.iloc[0]).rename(index=labels)
    contrib = contrib[contrib != 0].sort_values(ascending=False)
    money = contrib.map(lambda v: f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}")
    st.table(money.to_frame(name="adds to cost"))
    intercept = model.intercept_
    shown = f"-${-intercept:,.0f}" if intercept < 0 else f"${intercept:,.0f}"
    st.caption(f"The prediction also includes the model's starting value (intercept) of {shown}, which is not shown above.")

# Optional visualizations
if st.checkbox("Show dataset charge distribution (for context)"):
    df = pd.read_csv("insurance.csv")
    fig, ax = plt.subplots()
    sns.histplot(df["charges"], kde=True, ax=ax)
    st.pyplot(fig)
