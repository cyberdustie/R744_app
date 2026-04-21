import os
import io
import warnings
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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
import joblib
import warnings

warnings.filterwarnings("ignore")

# Page configuration and style
st.set_page_config(
    page_title="R-744 Surrogate Modeler",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: -0.03em;
    }
    /* Dark theme styling omitted for brevity, keep if needed */
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants
CANDIDATE_INPUTS = ["Dry Bulb Temperature", "Wet Bulb Temperature", "Building Load", "RSH", "RSC"]
CANDIDATE_OUTPUTS = ["W_comp", "P_gc", "P_e", "m_s", "m_rs", "m_rp"]
PRIMARY_TARGET = "P_gc"
LOG_FILE = "prediction_log.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 3
N_ITER = 15

MODEL_NAMES = [
    "Linear Regression", "Polynomial Regression", "SVR", "KNN",
    "Decision Tree", "Random Forest", "XGBoost", "LightGBM",
    "CatBoost", "ANN (MLP)",
]

# Helper functions to load data, preprocess, build models, evaluate, etc.
@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = {}
    for name in xl.sheet_names:
        df = xl.parse(name)
        df.columns = df.columns.str.strip()
        sheets[name] = df
    return sheets

def detect_columns(df):
    cols = set(df.columns)
    inputs = [c for c in CANDIDATE_INPUTS if c in cols]
    outputs = [c for c in CANDIDATE_OUTPUTS if c in cols]
    return inputs, outputs

@st.cache_data(show_spinner=False)
def preprocess(df_json, input_cols, output_col):
    df = pd.read_json(io.StringIO(df_json))
    cols = list(input_cols) + [output_col]
    df_clean = df[cols].dropna().reset_index(drop=True)
    X = df_clean[input_cols].values
    y = df_clean[[output_col]].values
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y).ravel()
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y_s, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return X_tr, X_te, y_tr, y_te, scaler_X, scaler_y, list(input_cols)

def build_estimator(model_name):
    M = {
        "Linear Regression": (
            Pipeline([("sc", StandardScaler(with_std=False)), ("lr", LinearRegression())]),
            {"lr__fit_intercept": [True, False]}
        ),
        "Polynomial Regression": (
            Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("lr", LinearRegression())]),
            {"poly__degree": [2, 3], "lr__fit_intercept": [True, False]}
        ),
        "SVR": (
            SVR(kernel="rbf"),
            {"C": [0.1, 1, 10, 50, 100], "gamma": ["scale", "auto", 0.01, 0.1], "epsilon": [0.01, 0.05, 0.1, 0.5]}
        ),
        "KNN": (
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7, 10, 15], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]}
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"max_depth": [None, 5, 10, 15], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 4]}
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2"]}
        ),
        "XGBoost": (
            xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0, tree_method="hist"),
            {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1, 0.2], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0]}
        ),
        "LightGBM": (
            lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1),
            {"n_estimators": [100, 200, 300], "max_depth": [-1, 5, 10], "learning_rate": [0.01, 0.05, 0.1, 0.2], "num_leaves": [31, 63, 127], "subsample": [0.7, 0.85, 1.0]}
        ),
        "CatBoost": (
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            {"iterations": [100, 200, 300], "depth": [4, 6, 8], "learning_rate": [0.01, 0.05, 0.1, 0.2], "l2_leaf_reg": [1, 3, 5]}
        ),
        "ANN (MLP)": (
            MLPRegressor(max_iter=600, random_state=RANDOM_STATE, early_stopping=True, n_iter_no_change=20),
            {"hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64, 32), (256, 128, 64)],
             "activation": ["relu", "tanh"],
             "alpha": [1e-4, 1e-3, 1e-2],
             "learning_rate_init": [1e-3, 5e-4, 1e-4]}
        ),
    }
    return M.get(model_name)

def evaluate_model(model, X_test, y_test, scaler_y):
    y_pred_s = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    errors = y_true - y_pred
    return {"RMSE": rmse, "MSE": mse, "R2": r2, "y_true": y_true, "y_pred": y_pred, "errors": errors}

def train_model(estimator, X_train, y_train):
    estimator.fit(X_train, y_train)
    return estimator

def optimize_model(estimator, param_grid, X_train, y_train):
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, n_iter=N_ITER, scoring="neg_root_mean_squared_error", cv=kf, random_state=RANDOM_STATE, n_jobs=-1, refit=True, error_score="raise")
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

def get_feature_importance(model, model_name, feature_names):
    try:
        if model_name in ("Random Forest", "Decision Tree", "XGBoost", "LightGBM"):
            imp = model.feature_importances_
        elif model_name == "CatBoost":
            imp = model.get_feature_importance()
        else:
            imp = None
    except:
        imp = None
    if imp is not None:
        imp = imp / (imp.sum() + 1e-12)
        return feature_names, imp
    return None, None

def log_prediction(sheet, model_name, target, input_vals, input_cols, prediction, m_before, m_after):
    row = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sheet": sheet,
        "Model": model_name,
        "Target": target,
        "Prediction": round(float(prediction), 5),
        "RMSE_before": round(m_before["RMSE"], 5),
        "RMSE_after": round(m_after["RMSE"], 5),
        "R2_after": round(m_after["R2"], 5),
    }
    for col, val in zip(input_cols, input_vals):
        row[f"input_{col}"] = round(float(val), 4)
    df_row = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        df_row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_FILE, mode="w", header=True, index=False)

# Main UI: minimal sidebar for upload and model selection
with st.sidebar:
    st.markdown("## R-744 Surrogate Modeler")
    uploaded = st.file_uploader("Upload Excel Workbook", type=["xlsx", "xls"])
    if uploaded:
        file_bytes = uploaded.read()
        sheets_dict = load_data(file_bytes)
        sheet_names = list(sheets_dict.keys())
        sheet_name = st.selectbox("Select sheet", sheet_names)
        df_sheet = sheets_dict[sheet_name]
        input_cols_all, output_cols_all = detect_columns(df_sheet)
        if input_cols_all and output_cols_all:
            target_col = st.selectbox("Target variable", output_cols_all)
            model_name = st.selectbox("Model", MODEL_NAMES)
            run_button = st.button("Train & Optimize")
        else:
            st.error("No recognized input/output columns.")
            st.stop()
    else:
        st.info("Please upload a file to proceed.")
        st.stop()

# Data preview
if 'df_sheet' in locals():
    st.write(f"Data preview for sheet: {sheet_name}")
    st.dataframe(df_sheet[input_cols_all + [target_col]].dropna().head(20))

# Training
if 'run_button' in locals() and run_button:
    df_json = df_sheet.to_json()
    X_tr, X_te, y_tr, y_te, sx, sy, feat_names = preprocess(df_json, tuple(input_cols_all), target_col)
    default_est, param_grid = build_estimator(model_name)
    fitted_default = train_model(default_est, X_tr, y_tr)
    m_before = evaluate_model(fitted_default, X_te, y_te, sy)
    try:
        best_est, best_params = optimize_model(fitted_default, param_grid, X_tr, y_tr)
        m_after = evaluate_model(best_est, X_te, y_te, sy)
    except:
        best_est, best_params = fitted_default, {}
        m_after = m_before
    # Store in session state
    st.session_state['m_before'] = m_before
    st.session_state['m_after'] = m_after
    st.session_state['best_model'] = best_est
    st.session_state['scaler_X'] = sx
    st.session_state['scaler_y'] = sy
    st.session_state['best_params'] = best_params
    st.session_state['input_cols'] = input_cols_all
    st.session_state['feature_names'] = feat_names
    st.success("Training complete!")

# Results display
if 'm_before' in st.session_state and 'm_after' in st.session_state:
    m_before = st.session_state['m_before']
    m_after = st.session_state['m_after']
    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("RMSE (Before)", f"{m_before['RMSE']:.4f}")
        st.metric("R² (Before)", f"{m_before['R2']:.4f}")

    with col2:
        st.metric("RMSE (After)", f"{m_after['RMSE']:.4f}")
        st.metric("R² (After)", f"{m_after['R2']:.4f}")

    # Optional: Show hyperparameters, logs, and plots as needed

# Prediction panel
if 'best_model' in st.session_state:
    st.markdown("### Real-Time Prediction")
    input_vals = []
    for feat in input_cols_all:
        col_data = df_sheet[feat].dropna()
        mn, mx = col_data.min(), col_data.max()
        med = col_data.median()
        step = (mx - mn) / 200 if (mx - mn) != 0 else 1e-3
        val = st.slider(feat, float(mn), float(mx), float(med), step=step)
        input_vals.append(val)
    X_user = st.session_state['scaler_X'].transform(np.array(input_vals).reshape(1, -1))
    pred_s = st.session_state['best_model'].predict(X_user)
    pred_val = float(st.session_state['scaler_y'].inverse_transform(pred_s.reshape(-1, 1)).ravel()[0])
    st.markdown(f"Predicted {target_col}: {pred_val:.4f}")

    # Log and download buttons omitted for brevity
