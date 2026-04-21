import os
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
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

st.set_page_config(page_title="R744 Model", layout="wide")

# -------------------------
# CONSTANTS
# -------------------------
INPUTS  = ["Dry Bulb Temperature", "Wet Bulb Temperature",
           "Building Load", "RSH", "RSC"]

OUTPUTS = ["W_comp", "P_gc", "P_e", "m_s", "m_rs", "m_rp"]

LOG_FILE = "prediction_log.csv"

# -------------------------
# LOAD DATA
# -------------------------
def load_excel(file):
    xl = pd.ExcelFile(file)
    return {s: xl.parse(s) for s in xl.sheet_names}

def detect(df):
    inputs = [c for c in INPUTS if c in df.columns]
    outputs = [c for c in OUTPUTS if c in df.columns]
    return inputs, outputs

# -------------------------
# PREPROCESS
# -------------------------
def preprocess(df, inputs, target):
    df = df[inputs + [target]].dropna()

    X = df[inputs].values
    y = df[[target]].values

    sx, sy = StandardScaler(), StandardScaler()
    Xs = sx.fit_transform(X)
    ys = sy.fit_transform(y).ravel()

    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42)

    return Xtr, Xte, ytr, yte, sx, sy

# -------------------------
# MODELS
# -------------------------
def get_model(name):
    models = {
        "Linear Regression": (
            Pipeline([("lr", LinearRegression())]),
            {"lr__fit_intercept": [True, False]}
        ),

        "Polynomial": (
            Pipeline([
                ("poly", PolynomialFeatures()),
                ("lr", LinearRegression())
            ]),
            {"poly__degree": [2, 3]}
        ),

        "SVR": (
            SVR(),
            {"C":[1,10,100], "epsilon":[0.1,0.5]}
        ),

        "KNN": (
            KNeighborsRegressor(),
            {"n_neighbors":[3,5,7]}
        ),

        "Decision Tree": (
            DecisionTreeRegressor(),
            {"max_depth":[None,5,10]}
        ),

        "Random Forest": (
            RandomForestRegressor(),
            {"n_estimators":[100,200]}
        ),

        "XGBoost": (
            xgb.XGBRegressor(),
            {"n_estimators":[100,200]}
        ),

        "LightGBM": (
            lgb.LGBMRegressor(),
            {"n_estimators":[100,200]}
        ),

        "CatBoost": (
            CatBoostRegressor(verbose=0),
            {"iterations":[100,200]}
        ),

        "ANN": (
            MLPRegressor(max_iter=500),
            {"hidden_layer_sizes":[(64,), (128,)]}
        )
    }
    return models[name]

# -------------------------
# TRAIN
# -------------------------
def train_model(model, params, X, y):
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        model, params, n_iter=10,
        cv=cv, scoring="r2", n_jobs=-1
    )

    search.fit(X, y)
    return search.best_estimator_

# -------------------------
# METRICS
# -------------------------
def evaluate(model, X, y, scaler_y):
    pred = model.predict(X)
    pred = scaler_y.inverse_transform(pred.reshape(-1,1)).ravel()
    y_true = scaler_y.inverse_transform(y.reshape(-1,1)).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, pred))
    r2 = r2_score(y_true, pred)

    return rmse, r2

# -------------------------
# UI
# -------------------------
st.title("R-744 Surrogate Model")

file = st.file_uploader("Upload Excel")

if file:
    sheets = load_excel(file)
    sheet = st.selectbox("Sheet", list(sheets.keys()))

    df = sheets[sheet]

    inputs, outputs = detect(df)

    target = st.selectbox("Target", outputs)
    model_name = st.selectbox("Model", list(get_model.__annotations__.keys()) if False else [
        "Linear Regression","Polynomial","SVR","KNN",
        "Decision Tree","Random Forest","XGBoost",
        "LightGBM","CatBoost","ANN"
    ])

    if st.button("Train"):
        Xtr, Xte, ytr, yte, sx, sy = preprocess(df, inputs, target)

        model, params = get_model(model_name)

        best = train_model(model, params, Xtr, ytr)

        rmse, r2 = evaluate(best, Xte, yte, sy)

        st.success("Training Done")
        st.write("RMSE:", rmse)
        st.write("R2:", r2)

        st.session_state["model"] = best
        st.session_state["sx"] = sx
        st.session_state["sy"] = sy
        st.session_state["inputs"] = inputs

# -------------------------
# PREDICTION
# -------------------------
if "model" in st.session_state:
    st.subheader("Prediction")

    vals = []
    for col in st.session_state["inputs"]:
        vals.append(st.number_input(col, value=0.0))

    if st.button("Predict"):
        X = np.array(vals).reshape(1,-1)
        Xs = st.session_state["sx"].transform(X)

        pred = st.session_state["model"].predict(Xs)
        pred = st.session_state["sy"].inverse_transform(pred.reshape(-1,1))[0][0]

        st.success(f"Prediction: {pred}")

        row = {"time": str(datetime.datetime.now()), "prediction": pred}
        pd.DataFrame([row]).to_csv(LOG_FILE, mode="a",
                                  header=not os.path.exists(LOG_FILE),
                                  index=False)

# -------------------------
# DOWNLOAD
# -------------------------
if "model" in st.session_state:
    buf = io.BytesIO()
    joblib.dump(st.session_state["model"], buf)
    buf.seek(0)

    st.download_button("Download Model", buf, "model.pkl")
