# CardioXAI: Heart Disease Risk Prediction Using Explainable AI

**Course:** Machine Learning Mini Project
**Author:** SAMIM (Roll No: 50)

CardioXAI predicts a patient's risk of heart disease from 11 clinical features and explains every prediction with SHAP, so the result is not a black box. The project includes a full ML pipeline (cleaning, EDA, three models, evaluation, explainability) and a Streamlit web app as a working prototype.

> Educational project only. Not a medical diagnosis tool.

---

## 1. Problem Statement and Objectives

Heart disease is a leading cause of death, and early risk screening can help doctors prioritise patients. Many ML models give a prediction without a reason, which makes them hard to trust in healthcare.

**Objectives**
1. Predict whether a patient has heart disease from routine clinical data.
2. Compare at least three ML algorithms using suitable metrics.
3. Choose a final model with a clear justification, giving priority to **Recall** (missing a sick patient is the costliest error).
4. Explain model decisions globally and per patient using SHAP.
5. Deliver a working prototype that shows the risk and the reasons behind it.

---

## 2. Dataset

**Heart Failure Prediction Dataset** by fedesoriano (Kaggle, September 2021).
Link: https://www.kaggle.com/fedesoriano/heart-failure-prediction

It combines five public heart disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, Stalog) into **918 patient records with 11 features** and one target.

| Feature | Meaning |
|---|---|
| Age | Age in years |
| Sex | M / F |
| ChestPainType | TA (typical angina), ATA (atypical angina), NAP (non-anginal pain), ASY (asymptomatic) |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| RestingECG | Normal, ST, LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y / N) |
| Oldpeak | ST depression during exercise |
| ST_Slope | Slope of peak exercise ST segment (Up, Flat, Down) |
| **HeartDisease** | **Target: 1 = heart disease, 0 = normal** |

**How to get the data:** download `heart.csv` from the Kaggle link above and place it in the project folder (or use the `heart.csv` included in this repository).

---

## 3. Methodology

1. **Data cleaning.** `info()` showed no missing values, but a manual check found hidden missing values stored as zero: **172 rows with Cholesterol = 0** and **1 row with RestingBP = 0**. These are impossible values, so they were replaced with the median. Output: `heart_clean.csv`.
2. **Exploratory data analysis.** Target distribution, age distribution, chest pain vs disease, correlation heatmap and group-wise disease rates.
3. **Preprocessing.** One-hot encoding of categorical columns (`drop_first=True`), 80/20 stratified train-test split (734 train / 184 test, `random_state=42`), standard scaling for Logistic Regression only.
4. **Models.** Logistic Regression, Random Forest (200 trees), XGBoost.
5. **Evaluation.** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, and 5-fold stratified cross-validation (Recall).
6. **Explainability.** SHAP `TreeExplainer` on the Random Forest: summary (beeswarm) plot, mean-|SHAP| bar plot and per-patient waterfall plot.
7. **Prototype.** Streamlit app with a risk gauge, top contributing factors in plain language, and a technical SHAP waterfall view.

---

## 4. Exploratory Data Analysis: Key Findings

Overall, **55.3%** of the patients have heart disease, so the classes are fairly balanced and accuracy is not misleading.

| Group | Disease rate |
|---|---|
| Chest pain ASY | **79.0%** |
| Chest pain TA | 43.5% |
| Chest pain NAP | 35.5% |
| Chest pain ATA | 13.9% |
| Exercise angina = Yes | **85.2%** |
| Exercise angina = No | 35.1% |
| ST slope Flat | **82.8%** |
| ST slope Down | 77.8% |
| ST slope Up | 19.7% |
| Fasting blood sugar > 120 = Yes | 79.4% |
| Fasting blood sugar > 120 = No | 48.0% |
| Male | 63.2% |
| Female | 25.9% |

**Averages by class**

| | Age | MaxHR | Oldpeak | Cholesterol | RestingBP |
|---|---|---|---|---|---|
| No disease | 50.6 | 148.2 | 0.4 | 238.7 | 130.2 |
| Disease | 55.9 | 127.7 | 1.3 | 246.9 | 134.4 |

**Observations**
- Asymptomatic (ASY) chest pain has the highest disease rate, which shows that heart disease can be "silent".
- Exercise angina and a Flat or Down ST slope are strongly associated with disease, while an Up slope is associated with a low rate.
- Patients with disease are older, reach a lower maximum heart rate and have a higher Oldpeak.
- Cholesterol differs only slightly between the groups, and about 19% of its values were imputed, so it is a weak separator here.
- The dataset has more male patients, so the sex difference may partly reflect sampling.
- These are correlations, not causation.

---

## 5. Results

### Test set (184 patients)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.891 | 0.894 | 0.912 | 0.903 | 0.933 |
| Random Forest | 0.853 | 0.871 | 0.863 | 0.867 | 0.931 |
| XGBoost | 0.859 | 0.880 | 0.863 | 0.871 | 0.914 |

### Confusion matrices (test set, 102 diseased patients)

| Model | True Neg | False Pos | False Neg (missed patients) | True Pos |
|---|---|---|---|---|
| Logistic Regression | 71 | 11 | 9 | 93 |
| Random Forest | 69 | 13 | 14 | 88 |
| XGBoost | 70 | 12 | 14 | 88 |

### 5-fold cross-validation (Recall)

| Model | Mean Recall | Std Dev |
|---|---|---|
| Logistic Regression | 0.884 | 0.040 |
| Random Forest | **0.908** | 0.026 |
| XGBoost | 0.874 | 0.025 |

### Model selection

On the single test split, Logistic Regression scored highest. However, the test set has only 184 patients, so one patient changes accuracy by about 0.5%. Cross-validation uses all 918 patients and is more reliable. There, **Random Forest has the highest Recall (0.908)**, and the gap to Logistic Regression is within one standard deviation, so the two models are close.

**Final model: Random Forest**, because of (1) the best cross-validated Recall, which matters most in a medical setting, (2) native compatibility with SHAP `TreeExplainer`, and (3) no need for a scaler in the app. Logistic Regression is kept as a strong, simple baseline.

---

## 6. Explainable AI (SHAP)

Most influential features (mean |SHAP|): **ST_Slope_Up (0.13)**, **ST_Slope_Flat (0.09)**, **ExerciseAngina_Y (0.07)**, **Oldpeak (0.05)** and **MaxHR (0.05)**.

- A Flat ST slope, exercise angina and a higher Oldpeak push the predicted risk **up**.
- An Up ST slope and a higher maximum heart rate push the risk **down**.
- Chest pain types ATA and NAP lower the risk relative to the ASY baseline.
- Age, Cholesterol and RestingBP have a comparatively small impact.

These findings agree with the group-wise rates in the EDA section.

---

## 7. Streamlit Prototype

The app (`app.py`) provides:
- Sidebar patient form with one-click demo patients (High / Mid / Low)
- Live risk gauge with a colour-coded level (Low < 40%, Moderate 40-70%, High >= 70%)
- Top 6 contributing factors in plain language
- Technical SHAP waterfall plot
- A Model Performance tab (metrics, cross-validation, SHAP summary, confusion matrices) and an About tab
- A warning for unrealistic inputs (for example a very high heart rate for the given age)

Example demo predictions: High-risk patient about 96%, mid patient about 68.5%, low-risk patient about 2%.

---

## 8. How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-run the experiments
#    Open Heart_Disease_Prediction.ipynb in VS Code or Jupyter and run all cells.
#    This regenerates heart_clean.csv, model.pkl, columns.pkl and model_results.csv.

# 4. Launch the app
streamlit run app.py
```

If `streamlit` is not recognised, use `python -m streamlit run app.py`.

---

## 9. Project Structure

```
heart-disease-prediction/
├── app.py                          # Streamlit prototype
├── Heart_Disease_Prediction.ipynb  # Full pipeline notebook
├── heart.csv                       # Original dataset
├── heart_clean.csv                 # Cleaned dataset
├── model.pkl                       # Trained Random Forest
├── columns.pkl                     # Feature column order for the app
├── model_results.csv               # Test-set comparison table
├── shap_summary.png                # SHAP summary plot
├── confusion_matrices.png          # Confusion matrices
├── .streamlit/config.toml          # App theme
├── requirements.txt
└── README.md
```

---

## 10. Limitations

- Small dataset (918 rows) and a small test set (184 patients), so results may not generalise to every population.
- About 19% of Cholesterol values were imputed with the median, and the median was computed on the full dataset before the train-test split (minor data leakage).
- Predictions may be unreliable for unrealistic input combinations outside the training distribution.
- SHAP explains model behaviour, it does not prove causation.
- The data comes from a few hospitals and is not validated for local populations.
- For educational purposes only. Not a medical diagnosis.

## 11. Future Work

- Hyperparameter tuning and a larger multi-source dataset
- Calibrated probabilities and external validation
- Imputation inside the cross-validation pipeline to remove leakage
- Deployment on Streamlit Community Cloud

---

## Acknowledgements

- Dataset: fedesoriano. (September 2021). *Heart Failure Prediction Dataset*. Kaggle. https://www.kaggle.com/fedesoriano/heart-failure-prediction
- Libraries: pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Matplotlib, Seaborn.
