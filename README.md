# Medical Insurance Cost Predictor

A fully deployed, end‑to‑end machine learning project that predicts annual medical insurance charges using linear regression. The project includes data exploration, preprocessing, model training, evaluation, deployment using Streamlit, and interpretability through feature contribution analysis.

---

## 🚀 Live Demo

**Live App:** *https://the-medical-cost-prediction.streamlit.app/*


## 📘 Project Overview

Insurance companies estimate medical charges using various personal factors. This project builds a simple but effective predictive model that:

* Learns patterns from historical medical cost data
* Predicts the expected annual medical charges for a new patient
* Provides clear feature‑level explanations of the prediction
* Runs as a clean, interactive Streamlit web app

---

## 🧵 Dataset

**Source:** Kaggle — Medical Cost Personal Dataset
**Rows:** 1,338 (1,337 after removing one duplicate)
**Features:**

* `age` — age of the patient
* `sex` — male/female
* `bmi` — body mass index
* `children` — dependents
* `smoker` — yes/no
* `region` — northeast, northwest, southeast, southwest
* `charges` — medical cost billed (target)

This dataset is noisy, skewed, and contains large real-world variability — making it a good test for regression.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)

Performed:

* Distribution of charges
* Scatter plots (age vs charges, BMI vs charges)
* Boxplots (smoker vs charges)
* Correlation heatmap

**Key Insights:**

* Smoking is the single largest cost driver
* Age shows moderate positive correlation
* BMI shows slight-to-moderate positive correlation
* Sex and region have minimal predictive impact

---

### 2. Preprocessing & Feature Engineering

Steps taken:

* Converted categorical variables (sex, smoker) into binary indicators
* One-hot encoded region
* Dropped original string columns
* Added `obese` (BMI 30+) and `smoker_obese` for the improved model

Final numerical features ready for model training.

---

### 3. Model Training

**Algorithm:** Linear Regression (scikit-learn)
**Train/Test Split:** 80/20

The model learns simple linear relationships between input features and charges. Coefficients are easy to interpret, making the model transparent.

---

## 📈 Model Performance

Baseline model (predicting mean cost only):

* **MAE:** ~9861.80

First linear regression model (original features):

* **MAE:** ~4177.04
* **RMSE:** ~5956.34
* **R²:** ~0.81

On this 80/20 split, it reduces MAE by **~58%** compared with the baseline. A 5-fold cross-validation gives a more cautious estimate: **R² ~0.74 (std ~0.04)** and **MAE ~4,210**.

### Improved model (used in the app)

The residuals showed the biggest misses were smokers with high BMI: smoking and obesity together cost far more than each one alone. Two features were added:

* `obese`: 1 if BMI is 30 or more
* `smoker_obese`: 1 if the person smokes **and** is obese

Same 5 folds for every model:

| Model | CV R² | CV MAE |
|---|---|---|
| Mean baseline | 0.00 | ~9,093 |
| Linear regression (original features) | 0.74 (std 0.04) | ~4,210 |
| **Linear + smoker × obese** | **0.86 (std 0.03)** | **~2,468** |
| Random forest | 0.85 | ~2,517 |
| Gradient boosting | 0.85 | ~2,518 |

On the same 80/20 split, the improved model gets **MAE ~2,369, RMSE ~4,142, R² ~0.91**, about **76%** lower error than the baseline. It beats both tree models while staying a linear regression, so every coefficient can still be read in dollars.

---

## 🔍 Feature Importance & Interpretation

Top contributors in the improved model (all else equal):

* **Smoking** — about **+$13.2k** for a smoker with BMI under 30, and about **+$32.8k** for a smoker with BMI 30+ (+$13.2k for smoking plus +$19.6k for smoking and obese together)
* **age** — about +$259 per year
* **children** — about +$559 per child
* **bmi** — a small effect once obesity and smoking are accounted for
* **sex / region** — small influence

In the first model, smoking added about +$23k on average, because that model could not tell obese smokers from the rest.

These match the patterns observed in EDA.

---

## 🖥 Web App (Streamlit)

The front-end is an interactive Streamlit application that allows users to:

* Enter patient details using sliders and dropdowns
* Generate predicted medical costs instantly
* View feature contributions for transparency
* Visualize dataset distributions (optional)

Run locally:

```
streamlit run app.py
```

---

## 🧪 How to Run Locally

1. Clone the repo:

```
git clone <repo-url>
cd medical_cost_prediction
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Launch Streamlit app:

```
streamlit run app.py
```

---

## 📉 Limitations

* Linear regression struggles with extreme outliers
* Model is limited by available features — missing health history, medications, income, etc.
* Predictions are point estimates without uncertainty bands (optional to add)
* Dataset may not generalize to real insurance pricing rules

---

## 📌 Future Improvements

* Tune the tree models further (they were compared with default-style settings)
* Use log-transformed target to reduce heteroscedasticity
* Add SHAP explanations
* Add batch CSV upload in the app
* Build a REST API version using FastAPI

---

## ⚖️ Ethical Note

This model is not for real medical underwriting.
Demographic features must be handled responsibly.
This project is strictly educational.

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* Kaggle dataset
* scikit-learn
* Streamlit
