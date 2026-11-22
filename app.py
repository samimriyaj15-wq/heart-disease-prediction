import os
import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

# ===== EDIT THESE =====
STUDENT_NAME = "SK SAMIM RIYAJ"
# ======================

st.set_page_config(page_title="CardioXAI | Heart Risk", page_icon="❤️", layout="wide")

# ---------- CSS ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 1.2rem; max-width: 1250px;}
#MainMenu, footer, .stDeployButton {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}

.hero {
  background: linear-gradient(120deg, #0f172a 0%, #1e293b 55%, #7f1d1d 100%);
  padding: 30px 34px; border-radius: 18px; margin-bottom: 22px;
}
.hero-title {font-size: 2.1rem; font-weight: 800; color: #ffffff; margin: 0;}
.hero-sub {color: #cbd5e1; font-size: 1rem; margin-top: 6px;}
.chips {margin-top: 16px;}
.chip {
  display: inline-block; padding: 5px 14px; margin: 0 8px 6px 0; border-radius: 999px;
  background: rgba(255,255,255,.12); color: #f1f5f9; font-size: .8rem; font-weight: 500;
  border: 1px solid rgba(255,255,255,.18);
}
.author {color: #94a3b8; font-size: .82rem; margin-top: 12px;}

.card {
  border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
  padding: 22px 24px; background: rgba(128,128,128,.06);
}
.card-title {font-size: .75rem; letter-spacing: .08em; font-weight: 700; opacity: .65; margin-bottom: 8px;}
.badge {
  display: inline-block; padding: 5px 18px; border-radius: 999px;
  color: #ffffff; font-weight: 700; font-size: .9rem;
}
.center {text-align: center;}
.note {font-size: .8rem; opacity: .7; margin-top: 14px;}

.factor {margin-bottom: 16px;}
.f-top {display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: 5px;}
.f-dir {font-weight: 600; font-size: .82rem;}
.f-bar {background: rgba(128,128,128,.2); border-radius: 999px; height: 9px; overflow: hidden;}
.f-bar div {height: 100%; border-radius: 999px;}
.summary {
  margin-top: 6px; padding: 12px 14px; border-radius: 10px; font-size: .88rem;
  background: rgba(128,128,128,.10);
}
.footer {text-align: center; font-size: .8rem; opacity: .6; margin-top: 30px;}
</style>
""",
    unsafe_allow_html=True,
)


def html(s):
    """Strip indentation so Markdown does not turn the HTML into a code block."""
    cleaned = "\n".join(l.strip() for l in s.splitlines() if l.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# ---------- Model ----------
@st.cache_resource
def load():
    model = joblib.load("model.pkl")
    cols = joblib.load("columns.pkl")
    explainer = shap.TreeExplainer(model)
    return model, cols, explainer


model, cols, explainer = load()

PREFIXES = ["ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope", "Sex"]


def base_name(col):
    for p in PREFIXES:
        if col.startswith(p + "_"):
            return p
    return col


def risk_level(p):
    if p >= 0.7:
        return "HIGH RISK", "#dc2626"
    if p >= 0.4:
        return "MODERATE RISK", "#f59e0b"
    return "LOW RISK", "#16a34a"


def gauge_svg(p, color):
    arc = 251.33  # half circle length for radius 80
    return f"""
<svg viewBox="0 0 200 120" width="100%" style="max-width:300px">
<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="rgba(128,128,128,.25)" stroke-width="16" stroke-linecap="round"/>
<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" stroke-dasharray="{arc * p:.1f} {arc}"/>
<text x="100" y="86" text-anchor="middle" font-size="34" font-weight="800" fill="{color}">{p * 100:.0f}%</text>
<text x="100" y="110" text-anchor="middle" font-size="9" font-weight="600" fill="gray" letter-spacing="1.5">RISK SCORE</text>
</svg>
"""


# ---------- Demo patients ----------
PRESETS = {
    "high": dict(age=60, sex="M", cp="ASY", bp=150, chol=260, fbs=1, ecg="Normal",
                 hr=110, ang="Y", oldpeak=2.5, slope="Flat"),
    "mid": dict(age=52, sex="M", cp="NAP", bp=135, chol=230, fbs=0, ecg="Normal",
                hr=140, ang="N", oldpeak=1.0, slope="Flat"),
    "low": dict(age=35, sex="F", cp="ATA", bp=120, chol=190, fbs=0, ecg="Normal",
                hr=170, ang="N", oldpeak=0.0, slope="Up"),
}
for k, v in PRESETS["mid"].items():
    st.session_state.setdefault(k, v)


def load_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# ---------- Hero ----------
html(f"""
<div class="hero">
<div class="hero-title">❤️ CardioXAI</div>
<div class="hero-sub">Explainable heart disease risk prediction. See the risk, and understand <b>why</b>.</div>
<div class="chips">
<span class="chip">Random Forest</span>
<span class="chip">SHAP Explainability</span>
<span class="chip">918 patients</span>
<span class="chip">11 clinical features</span>
</div>
<div class="author">Developed by <b>{STUDENT_NAME}</b></div>
</div>
""")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🩺 Patient Details")
    st.caption("Quick demo patients")
    b1, b2, b3 = st.columns(3)
    b1.button("🔴 High", on_click=load_preset, args=("high",))
    b2.button("🟠 Mid", on_click=load_preset, args=("mid",))
    b3.button("🟢 Low", on_click=load_preset, args=("low",))
    st.divider()

    st.markdown("**Demographics**")
    age = st.slider("Age", 20, 90, key="age")
    sex = st.selectbox("Sex", ["M", "F"], key="sex",
                       format_func=lambda x: "Male" if x == "M" else "Female")

    st.markdown("**Vitals & Lab**")
    bp = st.number_input("Resting BP (mm Hg)", 80, 220, key="bp")
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, key="chol")
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], key="fbs",
                       format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("**ECG & Exercise**")
    cp = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"], key="cp",
                      help="ASY: Asymptomatic, ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina")
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], key="ecg")
    hr = st.slider("Max Heart Rate", 60, 210, key="hr")
    ang = st.selectbox("Exercise Angina", ["N", "Y"], key="ang",
                       format_func=lambda x: "Yes" if x == "Y" else "No")
    oldpeak = st.number_input("Oldpeak (ST depression)", -3.0, 7.0, step=0.1, key="oldpeak")
    slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key="slope")

# ---------- Prediction ----------
row = dict.fromkeys(cols, 0)
row.update({"Age": age, "RestingBP": bp, "Cholesterol": chol,
            "FastingBS": fbs, "MaxHR": hr, "Oldpeak": oldpeak})
for key in [f"Sex_{sex}", f"ChestPainType_{cp}", f"RestingECG_{ecg}",
            f"ExerciseAngina_{ang}", f"ST_Slope_{slope}"]:
    if key in row:
        row[key] = 1

X_new = pd.DataFrame([row])[cols].astype(float)
prob = float(model.predict_proba(X_new)[0, 1])
label, color = risk_level(prob)

sv = explainer(X_new)
sv1 = sv[:, :, 1] if sv.values.ndim == 3 else sv

agg = {}
for c, v in zip(cols, sv1.values[0]):
    agg[base_name(c)] = agg.get(base_name(c), 0.0) + float(v)

disp = {
    "Age": ("Age", f"{age} yrs"),
    "Sex": ("Sex", "Male" if sex == "M" else "Female"),
    "ChestPainType": ("Chest pain type", cp),
    "RestingBP": ("Resting BP", f"{bp} mm Hg"),
    "Cholesterol": ("Cholesterol", f"{chol} mg/dl"),
    "FastingBS": ("Fasting sugar > 120", "Yes" if fbs else "No"),
    "RestingECG": ("Resting ECG", ecg),
    "MaxHR": ("Max heart rate", f"{hr} bpm"),
    "ExerciseAngina": ("Exercise angina", "Yes" if ang == "Y" else "No"),
    "Oldpeak": ("Oldpeak", f"{oldpeak}"),
    "ST_Slope": ("ST slope", slope),
}

top = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
maxv = max(abs(v) for _, v in top) or 1e-9

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Model Performance", "ℹ️ About"])

# ===== Tab 1: Prediction =====
with tab1:
    if hr > (220 - age) + 20:
        st.warning("Max Heart Rate looks unrealistic for this age. The prediction may be unreliable.")

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        html(f"""
<div class="card center">
<div class="card-title">PREDICTED HEART DISEASE RISK</div>
{gauge_svg(prob, color)}
<div style="margin-top:6px"><span class="badge" style="background:{color}">{label}</span></div>
<div class="note">Educational project. Not a medical diagnosis.</div>
</div>
""")

    with right:
        bars = ""
        for name, val in top:
            nice, shown = disp[name]
            col = "#dc2626" if val > 0 else "#16a34a"
            arrow = "▲ raises risk" if val > 0 else "▼ lowers risk"
            bars += f"""
<div class="factor">
<div class="f-top"><span>{nice}: <b>{shown}</b></span><span class="f-dir" style="color:{col}">{arrow}</span></div>
<div class="f-bar"><div style="width:{abs(val) / maxv * 100:.0f}%;background:{col}"></div></div>
</div>
"""
        ups = [(k, v) for k, v in agg.items() if v > 0]
        downs = [(k, v) for k, v in agg.items() if v < 0]
        parts = []
        if ups:
            k = max(ups, key=lambda kv: kv[1])[0]
            parts.append(f"Biggest risk-raising factor: <b>{disp[k][0]} ({disp[k][1]})</b>.")
        if downs:
            k = min(downs, key=lambda kv: kv[1])[0]
            parts.append(f"Biggest risk-lowering factor: <b>{disp[k][0]} ({disp[k][1]})</b>.")
        summary = " ".join(parts)

        html(f"""
<div class="card">
<div class="card-title">WHY THIS PREDICTION? TOP 6 FACTORS</div>
{bars}
<div class="summary">{summary}</div>
</div>
""")

    st.write("")
    with st.expander("🔬 Technical view: SHAP waterfall plot"):
        st.caption("Red = pushes risk up, Blue = pushes risk down. f(x) is the final predicted probability.")
        fig = plt.figure()
        shap.plots.waterfall(sv1[0], show=False)
        st.pyplot(fig)
        plt.close(fig)

# ===== Tab 2: Model Performance =====
with tab2:
    st.subheader("Model Comparison")
    if os.path.exists("model_results.csv"):
        res = pd.read_csv("model_results.csv")
        rf_row = res[res["Model"] == "Random Forest"]
        if not rf_row.empty:
            rf_row = rf_row.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy (RF)", f"{rf_row['Accuracy']:.3f}")
            m2.metric("Recall (RF)", f"{rf_row['Recall']:.3f}")
            m3.metric("F1 (RF)", f"{rf_row['F1']:.3f}")
            m4.metric("ROC-AUC (RF)", f"{rf_row['ROC-AUC']:.3f}")
        st.dataframe(res, hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot(
            kind="bar", ax=ax, rot=0
        )
        ax.set_ylim(0.8, 1.0)
        ax.set_ylabel("Score (y-axis starts at 0.8)")
        ax.legend(loc="lower right", ncol=5, fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("model_results.csv not found in the project folder.")

    st.subheader("5-Fold Cross-Validation (Recall)")
    st.dataframe(
        pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Mean Recall": [0.884, 0.908, 0.874],
            "Std Dev": [0.040, 0.026, 0.025],
        }),
        hide_index=True,
    )
    st.info(
        "Final model: **Random Forest**. It achieved the highest cross-validation recall (0.908) "
        "and works naturally with SHAP TreeExplainer. Logistic Regression is a strong baseline, "
        "and the difference between the two is small."
    )

    c1, c2 = st.columns(2)
    if os.path.exists("shap_summary.png"):
        c1.markdown("**Global Feature Impact (SHAP)**")
        c1.image("shap_summary.png")
    if os.path.exists("confusion_matrices.png"):
        c2.markdown("**Confusion Matrices**")
        c2.image("confusion_matrices.png")

# ===== Tab 3: About =====
with tab3:
    st.subheader("About this project")
    st.write(
        "CardioXAI predicts a patient's risk of heart disease from 11 clinical features "
        "and explains every prediction using SHAP, so the result is not a black box."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients", "918")
    k2.metric("Features", "11")
    k3.metric("Models compared", "3")
    k4.metric("Final model", "Random Forest")

    st.write("")
    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("#### Dataset")
        st.markdown(
            """
**Heart Failure Prediction Dataset** by fedesoriano (Kaggle, 2021).
It combines five public heart disease datasets (Cleveland, Hungarian, Switzerland,
Long Beach VA, Stalog) into 918 patient records with 11 common features.
Target: `HeartDisease` (1 = disease, 0 = normal).
"""
        )

        st.markdown("#### Methodology")
        st.markdown(
            """
1. Data cleaning: zero values in Cholesterol and RestingBP treated as missing, filled with the median
2. Exploratory data analysis and visualization
3. One-hot encoding and an 80/20 stratified train-test split
4. Training of Logistic Regression, Random Forest and XGBoost
5. Evaluation with Accuracy, Precision, Recall, F1, ROC-AUC and 5-fold cross-validation
6. Explainability with SHAP (global and per-patient)
7. Streamlit prototype
"""
        )

    with b:
        st.markdown("#### Input features")
        st.dataframe(
            pd.DataFrame({
                "Feature": ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
                            "Oldpeak", "ST_Slope"],
                "Meaning": [
                    "Age in years",
                    "Male / Female",
                    "TA, ATA, NAP, ASY",
                    "Resting blood pressure (mm Hg)",
                    "Serum cholesterol (mg/dl)",
                    "Fasting blood sugar > 120 mg/dl",
                    "Resting ECG result (Normal, ST, LVH)",
                    "Maximum heart rate achieved",
                    "Exercise-induced angina (Y/N)",
                    "ST depression during exercise",
                    "Slope of peak exercise ST segment",
                ],
            }),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("#### Limitations")
    st.markdown(
        """
- Small dataset (918 rows) and a small test set (184 patients), so results may not generalize to every population.
- About 19% of Cholesterol values were imputed with the median, and the median was computed on the full dataset before the train-test split (minor data leakage).
- Predictions may be unreliable for unrealistic input combinations.
- SHAP explains model behaviour, it does not prove causation.
- For educational purposes only. Not a medical diagnosis.
"""
    )

    st.caption(
        "Dataset citation: fedesoriano. (September 2021). Heart Failure Prediction Dataset. "
        "Retrieved from https://www.kaggle.com/fedesoriano/heart-failure-prediction"
    )

html(f"""
<div class="footer">CardioXAI &nbsp;|&nbsp; Developed by {STUDENT_NAME}</div>
""")