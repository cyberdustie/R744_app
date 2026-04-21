import os
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

st.set_page_config(layout="wide", page_title="R744 Animated UI")

# -----------------------------
# ADVANCED GLASS + ANIMATIONS
# -----------------------------
st.markdown("""
<style>

/* Background gradient animation */
body {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c2b);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    color: white;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    transition: all 0.3s ease;
}

/* Hover glow */
.glass:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 0 25px rgba(88,166,255,0.35);
    border: 1px solid rgba(88,166,255,0.6);
}

/* Metric */
.metric {
    font-size: 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

/* Fade-in animation */
.fade-in {
    animation: fadeIn 0.8s ease-in-out;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Buttons glow */
.stButton > button {
    background: rgba(88,166,255,0.2);
    border: 1px solid rgba(88,166,255,0.4);
    color: white;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    background: rgba(88,166,255,0.4);
    box-shadow: 0 0 15px rgba(88,166,255,0.6);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# CONSTANTS
# -----------------------------
INPUTS  = ["Dry Bulb Temperature","Wet Bulb Temperature","Building Load","RSH","RSC"]
OUTPUTS = ["W_comp","P_gc","P_e","m_s","m_rs","m_rp"]

# -----------------------------
# MODELS
# -----------------------------
def get_model(name):
    return {
        "Linear": (LinearRegression(), {}),
        "Polynomial": (
            Pipeline([("poly", PolynomialFeatures()), ("lr", LinearRegression())]),
            {"poly__degree":[2,3]}
        ),
        "SVR": (SVR(), {"C":[1,10,100]}),
        "KNN": (KNeighborsRegressor(), {"n_neighbors":[3,5,7]}),
        "Decision Tree": (DecisionTreeRegressor(), {"max_depth":[None,5,10]}),
        "Random Forest": (RandomForestRegressor(), {"n_estimators":[100,200]}),
        "XGBoost": (xgb.XGBRegressor(), {"n_estimators":[100,200]}),
        "LightGBM": (lgb.LGBMRegressor(), {"n_estimators":[100,200]}),
        "CatBoost": (CatBoostRegressor(verbose=0), {"iterations":[100,200]}),
        "ANN": (MLPRegressor(max_iter=500), {"hidden_layer_sizes":[(64,), (128,)]})
    }[name]

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 class='fade-in'>❄️ R-744 Animated Dashboard</h1>", unsafe_allow_html=True)

file = st.file_uploader("Upload Excel")

if file:
    df = pd.read_excel(file)

    inputs = [c for c in INPUTS if c in df.columns]
    outputs = [c for c in OUTPUTS if c in df.columns]

    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Target", outputs)
    with col2:
        model_name = st.selectbox("Model", [
            "Linear","Polynomial","SVR","KNN","Decision Tree",
            "Random Forest","XGBoost","LightGBM","CatBoost","ANN"
        ])

    if st.button("Train Model"):

        df = df[inputs + [target]].dropna()

        X = df[inputs].values
        y = df[[target]].values

        sx, sy = StandardScaler(), StandardScaler()
        Xs = sx.fit_transform(X)
        ys = sy.fit_transform(y).ravel()

        Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2)

        model, params = get_model(model_name)

        search = RandomizedSearchCV(model, params, n_iter=10,
                                    cv=KFold(3), n_jobs=-1)
        search.fit(Xtr, ytr)

        best = search.best_estimator_

        pred = best.predict(Xte)

        y_true = sy.inverse_transform(yte.reshape(-1,1)).ravel()
        y_pred = sy.inverse_transform(pred.reshape(-1,1)).ravel()

        st.session_state.update({
            "model": best,
            "sx": sx,
            "sy": sy,
            "inputs": inputs,
            "y_true": y_true,
            "y_pred": y_pred,
            "errors": y_true - y_pred
        })

        st.success("Model Ready")

# -----------------------------
# DASHBOARD
# -----------------------------
if "model" in st.session_state:

    tabs = st.tabs(["📊 Metrics", "📈 Charts", "🎯 Prediction"])

    # METRICS
    with tabs[0]:
        rmse = np.sqrt(mean_squared_error(
            st.session_state["y_true"], st.session_state["y_pred"]))
        r2 = r2_score(
            st.session_state["y_true"], st.session_state["y_pred"])

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"""
            <div class="glass fade-in">
                <div>RMSE</div>
                <div class="metric">{rmse:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="glass fade-in">
                <div>R² Score</div>
                <div class="metric">{r2:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    # CHARTS
    with tabs[1]:
        fig1, ax1 = plt.subplots()
        ax1.scatter(st.session_state["y_true"], st.session_state["y_pred"])
        ax1.set_title("Actual vs Predicted")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.hist(st.session_state["errors"], bins=25)
        ax2.set_title("Residuals")
        st.pyplot(fig2)

        model = st.session_state["model"]
        if hasattr(model, "feature_importances_"):
            fig3, ax3 = plt.subplots()
            imp = model.feature_importances_
            idx = np.argsort(imp)
            ax3.barh(np.array(st.session_state["inputs"])[idx], imp[idx])
            ax3.set_title("Feature Importance")
            st.pyplot(fig3)

    # PREDICTION
    with tabs[2]:
        st.markdown('<div class="glass fade-in">', unsafe_allow_html=True)

        vals = []
        for col in st.session_state["inputs"]:
            vals.append(st.number_input(col, value=0.0))

        if st.button("Predict"):
            X = np.array(vals).reshape(1,-1)
            Xs = st.session_state["sx"].transform(X)

            pred = st.session_state["model"].predict(Xs)
            pred = st.session_state["sy"].inverse_transform(pred.reshape(-1,1))[0][0]

            st.success(f"Prediction: {pred:.4f}")

        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# DOWNLOAD
# -----------------------------
if "model" in st.session_state:
    buf = io.BytesIO()
    joblib.dump(st.session_state["model"], buf)
    buf.seek(0)

    st.download_button("⬇ Download Model", buf, "model.pkl")
