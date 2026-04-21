"""
=============================================================================
  R-744 HYBRID SYSTEM — SURROGATE MODELING DASHBOARD
  Streamlit App | ML Pipeline | Real-Time Prediction | Experiment Tracking
=============================================================================
"""

import os
import io
import warnings
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
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
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

"""─────────────────────────────────────────────────────────────────────────"""
"""  PAGE CONFIG & GLOBAL STYLE                                             """
"""─────────────────────────────────────────────────────────────────────────"""

st.set_page_config(
    page_title="R-744 Surrogate Modeler",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: -0.03em;
  }

  /* Dark engineering theme */
  .main { background: #0d1117; }
  .stApp { background: #0d1117; color: #c9d1d9; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
  }
  section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

  /* Metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
  }
  .metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #8b949e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.55rem;
    font-weight: 600;
    color: #58a6ff;
  }
  .metric-value.good  { color: #3fb950; }
  .metric-value.warn  { color: #d29922; }
  .metric-value.bad   { color: #f85149; }

  /* Section headers */
  .section-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    background: #1f6feb22;
    border: 1px solid #1f6feb55;
    color: #58a6ff;
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
  }

  /* Prediction result box */
  .pred-box {
    background: linear-gradient(135deg, #1a2744 0%, #0f1b30 100%);
    border: 1px solid #1f6feb;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
  }
  .pred-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .pred-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #79c0ff;
    line-height: 1.1;
  }

  /* Log table styling */
  .stDataFrame { border-radius: 6px; }

  /* Divider */
  hr { border-color: #21262d; margin: 1.2rem 0; }

  /* Button */
  .stButton > button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    padding: 8px 20px;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #388bfd; }

  /* Download button */
  .stDownloadButton > button {
    background: #238636;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
  }

  /* Tabs */
  .stTabs [role="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
  }

  /* Info boxes */
  .stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


"""─────────────────────────────────────────────────────────────────────────"""
"""  CONSTANTS                                                              """
"""─────────────────────────────────────────────────────────────────────────"""

CANDIDATE_INPUTS  = ["Dry Bulb Temperature", "Wet Bulb Temperature",
                     "Building Load", "RSH", "RSC"]
CANDIDATE_OUTPUTS = ["W_comp", "P_gc", "P_e", "m_s", "m_rs", "m_rp"]
PRIMARY_TARGET    = "P_gc"
LOG_FILE          = "prediction_log.csv"
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
CV_FOLDS          = 3
N_ITER            = 15

MODEL_NAMES = [
    "Linear Regression", "Polynomial Regression", "SVR", "KNN",
    "Decision Tree", "Random Forest", "XGBoost", "LightGBM",
    "CatBoost", "ANN (MLP)",
]

"""  Matplotlib dark style for all plots """
plt.style.use("dark_background")
PLOT_BG   = "#161b22"
PLOT_FG   = "#c9d1d9"
ACCENT_B  = "#58a6ff"
ACCENT_O  = "#f0883e"
ACCENT_G  = "#3fb950"
ACCENT_R  = "#f85149"


"""─────────────────────────────────────────────────────────────────────────"""
"""  DATA FUNCTIONS                                                         """
"""─────────────────────────────────────────────────────────────────────────"""

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> dict:
    """
    Parse the uploaded Excel workbook.
    Returns dict { sheet_name: DataFrame }.
    """
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = {}
    for name in xl.sheet_names:
        df = xl.parse(name)
        df.columns = df.columns.str.strip()
        sheets[name] = df
    return sheets


def detect_columns(df: pd.DataFrame):
    """
    Auto-detect which candidate input/output columns exist in df.
    Returns (input_cols, output_cols).
    """
    cols = set(df.columns)
    inputs  = [c for c in CANDIDATE_INPUTS  if c in cols]
    outputs = [c for c in CANDIDATE_OUTPUTS if c in cols]
    return inputs, outputs


@st.cache_data(show_spinner=False)
def preprocess(df_json: str, input_cols: tuple, output_col: str):
    """
    Clean, split, and scale data for a single target.
    Returns (X_train, X_test, y_train, y_test, scaler_X, scaler_y, feature_names).
    Accepts df_json to be cache-key friendly.
    """
    df = pd.read_json(io.StringIO(df_json))
    cols = list(input_cols) + [output_col]
    df_clean = df[cols].dropna().reset_index(drop=True)

    X = df_clean[list(input_cols)].values
    y = df_clean[[output_col]].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X)
    y_s = scaler_y.fit_transform(y).ravel()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_s, y_s, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_tr, X_te, y_tr, y_te, scaler_X, scaler_y, list(input_cols)


"""─────────────────────────────────────────────────────────────────────────"""
"""  MODEL FACTORY                                                          """
"""─────────────────────────────────────────────────────────────────────────"""

def build_estimator(model_name: str):
    """
    Return (default_estimator, param_grid) for the given model name.
    """
    M = {
        "Linear Regression": (
            Pipeline([("sc", StandardScaler(with_std=False)),
                      ("lr", LinearRegression())]),
            {"lr__fit_intercept": [True, False]}
        ),
        "Polynomial Regression": (
            Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                      ("lr",   LinearRegression())]),
            {"poly__degree": [2, 3], "lr__fit_intercept": [True, False]}
        ),
        "SVR": (
            SVR(kernel="rbf"),
            {"C": [0.1, 1, 10, 50, 100],
             "gamma": ["scale", "auto", 0.01, 0.1],
             "epsilon": [0.01, 0.05, 0.1, 0.5]}
        ),
        "KNN": (
            KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7, 10, 15],
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"max_depth": [None, 5, 10, 15],
             "min_samples_split": [2, 5, 10],
             "min_samples_leaf": [1, 2, 4]}
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [100, 200, 300],
             "max_depth": [None, 10, 20],
             "min_samples_split": [2, 5, 10],
             "max_features": ["sqrt", "log2"]}
        ),
        "XGBoost": (
            xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0, tree_method="hist"),
            {"n_estimators": [100, 200, 300],
             "max_depth": [3, 5, 7],
             "learning_rate": [0.01, 0.05, 0.1, 0.2],
             "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.7, 0.85, 1.0]}
        ),
        "LightGBM": (
            lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, n_jobs=-1),
            {"n_estimators": [100, 200, 300],
             "max_depth": [-1, 5, 10],
             "learning_rate": [0.01, 0.05, 0.1, 0.2],
             "num_leaves": [31, 63, 127],
             "subsample": [0.7, 0.85, 1.0]}
        ),
        "CatBoost": (
            CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
            {"iterations": [100, 200, 300],
             "depth": [4, 6, 8],
             "learning_rate": [0.01, 0.05, 0.1, 0.2],
             "l2_leaf_reg": [1, 3, 5]}
        ),
        "ANN (MLP)": (
            MLPRegressor(max_iter=600, random_state=RANDOM_STATE,
                         early_stopping=True, n_iter_no_change=20),
            {"hidden_layer_sizes": [(64,), (128,), (64, 32),
                                    (128, 64, 32), (256, 128, 64)],
             "activation": ["relu", "tanh"],
             "alpha": [1e-4, 1e-3, 1e-2],
             "learning_rate_init": [1e-3, 5e-4, 1e-4]}
        ),
    }
    return M[model_name]


"""─────────────────────────────────────────────────────────────────────────"""
"""  TRAIN / OPTIMIZE / EVALUATE                                            """
"""─────────────────────────────────────────────────────────────────────────"""

def evaluate_model(model, X_test, y_test, scaler_y):
    """
    Predict on test set (inverse-transformed) and return metrics dict.
    """
    y_pred_s = model.predict(X_test)
    y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true   = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    errors = y_true - y_pred
    return {"RMSE": rmse, "MSE": mse, "R2": r2,
            "y_true": y_true, "y_pred": y_pred, "errors": errors}


def train_model(estimator, X_train, y_train):
    """Fit estimator and return fitted model."""
    estimator.fit(X_train, y_train)
    return estimator


def optimize_model(estimator, param_grid, X_train, y_train):
    """
    Run RandomizedSearchCV and return (best_estimator, best_params).
    """
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=kf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def get_feature_importance(model, model_name, feature_names):
    """
    Extract feature importances for tree/boosting models.
    Returns (names, values) or (None, None).
    """
    imp = None
    try:
        if model_name in ("Random Forest", "Decision Tree", "XGBoost", "LightGBM"):
            imp = model.feature_importances_
        elif model_name == "CatBoost":
            imp = model.get_feature_importance()
    except Exception:
        pass
    if imp is not None:
        imp = imp / (imp.sum() + 1e-12)
        return feature_names, imp
    return None, None


"""─────────────────────────────────────────────────────────────────────────"""
"""  PREDICTION LOGGING                                                     """
"""─────────────────────────────────────────────────────────────────────────"""

def log_prediction(sheet, model_name, target, input_vals,
                   input_cols, prediction, m_before, m_after):
    """
    Append one prediction record to the CSV log file.
    """
    row = {
        "Timestamp":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Sheet":       sheet,
        "Model":       model_name,
        "Target":      target,
        "Prediction":  round(float(prediction), 5),
        "RMSE_before": round(m_before["RMSE"], 5),
        "RMSE_after":  round(m_after["RMSE"], 5),
        "R2_after":    round(m_after["R2"], 5),
    }
    for col, val in zip(input_cols, input_vals):
        row[f"input_{col}"] = round(float(val), 4)

    df_row = pd.DataFrame([row])
    if os.path.exists(LOG_FILE):
        df_row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_FILE, mode="w", header=True, index=False)


"""─────────────────────────────────────────────────────────────────────────"""
"""  VISUALIZATION                                                          """
"""─────────────────────────────────────────────────────────────────────────"""

def make_fig(nrows, ncols, h=3.5):
    """Helper to create a dark-themed figure."""
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.2, nrows * h),
                             facecolor=PLOT_BG)
    if not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1) if nrows == 1 else axes.reshape(-1, 1)
    for ax in axes.flat:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=PLOT_FG, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
    return fig, axes


def plot_before_after(m_before, m_after):
    """
    2×2 grid: RMSE bar | MSE bar | R² bar | Actual vs Predicted scatter.
    Returns a compact matplotlib figure.
    """
    fig, axes = make_fig(2, 2, h=3.0)

    labels  = ["Before", "After"]
    colors  = [ACCENT_B, ACCENT_G]

    """ RMSE """
    ax = axes[0, 0]
    vals = [m_before["RMSE"], m_after["RMSE"]]
    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="none")
    ax.set_title("RMSE", color=PLOT_FG, fontsize=8, fontweight="bold")
    ax.set_ylabel("Value", color=PLOT_FG, fontsize=7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v * 1.02,
                f"{v:.4f}", ha="center", va="bottom",
                color=PLOT_FG, fontsize=7)
    ax.grid(axis="y", alpha=0.15, color=PLOT_FG)

    """ MSE """
    ax = axes[0, 1]
    vals = [m_before["MSE"], m_after["MSE"]]
    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="none")
    ax.set_title("MSE", color=PLOT_FG, fontsize=8, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v * 1.02,
                f"{v:.4f}", ha="center", va="bottom",
                color=PLOT_FG, fontsize=7)
    ax.grid(axis="y", alpha=0.15, color=PLOT_FG)

    """ R² """
    ax = axes[1, 0]
    vals = [m_before["R2"], m_after["R2"]]
    bars = ax.bar(labels, vals, color=colors, width=0.45, edgecolor="none")
    ax.set_title("R² Score", color=PLOT_FG, fontsize=8, fontweight="bold")
    ax.set_ylim(min(0, min(vals) - 0.05), 1.05)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01,
                f"{v:.4f}", ha="center", va="bottom",
                color=PLOT_FG, fontsize=7)
    ax.grid(axis="y", alpha=0.15, color=PLOT_FG)

    """ Actual vs Predicted (after) """
    ax = axes[1, 1]
    y_true  = m_after["y_true"]
    y_pred  = m_after["y_pred"]
    ax.scatter(y_true, y_pred, s=12, alpha=0.55,
               color=ACCENT_B, edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color=ACCENT_R, linewidth=1, linestyle="--")
    ax.set_title("Actual vs Predicted", color=PLOT_FG, fontsize=8, fontweight="bold")
    ax.set_xlabel("Actual", color=PLOT_FG, fontsize=7)
    ax.set_ylabel("Predicted", color=PLOT_FG, fontsize=7)
    ax.grid(alpha=0.1, color=PLOT_FG)

    fig.tight_layout(pad=1.2)
    return fig


def plot_error_distribution(errors):
    """Compact histogram of prediction residuals."""
    fig, axes = make_fig(1, 1, h=3.0)
    ax = axes[0, 0]
    ax.hist(errors, bins=25, color=ACCENT_B, alpha=0.75,
            edgecolor="#0d1117", linewidth=0.4)
    ax.axvline(0, color=ACCENT_R, linewidth=1, linestyle="--")
    ax.set_title("Residual Distribution (After Optimization)",
                 color=PLOT_FG, fontsize=8, fontweight="bold")
    ax.set_xlabel("Error (Actual − Predicted)", color=PLOT_FG, fontsize=7)
    ax.set_ylabel("Count", color=PLOT_FG, fontsize=7)
    ax.grid(axis="y", alpha=0.15, color=PLOT_FG)
    fig.tight_layout(pad=1.2)
    return fig


def plot_feature_importance(names, importances, model_name):
    """Horizontal bar chart of feature importances."""
    fig, axes = make_fig(1, 1, h=max(2.5, 0.55 * len(names)))
    ax = axes[0, 0]
    idx = np.argsort(importances)
    colors_bar = plt.cm.viridis(np.linspace(0.25, 0.85, len(names)))
    ax.barh([names[i] for i in idx],
            [importances[i] for i in idx],
            color=colors_bar, edgecolor="none", height=0.55)
    ax.set_title(f"Feature Importance — {model_name}",
                 color=PLOT_FG, fontsize=8, fontweight="bold")
    ax.set_xlabel("Normalized Importance", color=PLOT_FG, fontsize=7)
    ax.grid(axis="x", alpha=0.15, color=PLOT_FG)
    fig.tight_layout(pad=1.2)
    return fig


def plot_model_comparison(log_df, target):
    """
    Bar chart comparing RMSE_after and R2_after across models
    logged for the current target.
    """
    sub = log_df[log_df["Target"] == target].drop_duplicates(
        subset=["Model"], keep="last"
    ).sort_values("RMSE_after")

    if sub.empty or len(sub) < 2:
        return None

    fig, axes = make_fig(1, 2, h=3.0)

    """ RMSE """
    ax = axes[0, 0]
    colors_m = plt.cm.plasma(np.linspace(0.2, 0.85, len(sub)))
    bars = ax.bar(sub["Model"], sub["RMSE_after"],
                  color=colors_m, edgecolor="none", width=0.55)
    ax.set_title(f"RMSE After — {target}", color=PLOT_FG, fontsize=8, fontweight="bold")
    ax.set_xticklabels(sub["Model"], rotation=35, ha="right", fontsize=6.5)
    ax.grid(axis="y", alpha=0.15, color=PLOT_FG)

    """ R² """
    ax = axes[0, 1]
    ax.bar(sub["Model"], sub["R2_after"],
           color=colors_m, edgecolor="none", width=0.55)
    ax.set_title(f"R² After — {target}", color=PLOT_FG, fontsize=8, fontweight="bold")
    ax.set_xticklabels(sub["Model"], rotation=35, ha="right", fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.15, color=PLOT_FG)

    fig.tight_layout(pad=1.2)
    return fig


"""─────────────────────────────────────────────────────────────────────────"""
"""  METRIC CARD HELPER                                                     """
"""─────────────────────────────────────────────────────────────────────────"""

def metric_card(label, value, color_class=""):
    """Render a styled metric card."""
    st.markdown(
        f"""<div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value {color_class}">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def r2_class(r2):
    if r2 >= 0.90: return "good"
    if r2 >= 0.75: return "warn"
    return "bad"


"""─────────────────────────────────────────────────────────────────────────"""
"""  STREAMLIT SESSION STATE INITIALISATION                                 """
"""─────────────────────────────────────────────────────────────────────────"""

for key in ["m_before", "m_after", "best_model",
            "scaler_X", "scaler_y", "best_params",
            "input_cols", "feature_names"]:
    if key not in st.session_state:
        st.session_state[key] = None


"""─────────────────────────────────────────────────────────────────────────"""
"""  SIDEBAR                                                                """
"""─────────────────────────────────────────────────────────────────────────"""

with st.sidebar:
    st.markdown("## ❄️ R-744 Modeler")
    st.markdown("---")

    uploaded = st.file_uploader("Upload Excel Workbook", type=["xlsx", "xls"])

    if uploaded is None:
        st.info("Upload your Excel file to begin.")
        st.stop()

    file_bytes = uploaded.read()
    sheets_dict = load_data(file_bytes)
    sheet_names = list(sheets_dict.keys())

    st.markdown("### Sheet")
    sheet_name = st.selectbox("Select sheet", sheet_names, label_visibility="collapsed")
    df_sheet = sheets_dict[sheet_name]

    input_cols_all, output_cols_all = detect_columns(df_sheet)

    if not input_cols_all or not output_cols_all:
        st.error("No recognised input/output columns on this sheet.")
        st.stop()

    st.markdown("### Target Variable")
    target_col = st.selectbox("Select target", output_cols_all, label_visibility="collapsed")

    st.markdown("### Model")
    model_name = st.selectbox("Select model", MODEL_NAMES, label_visibility="collapsed")

    st.markdown("---")
    run_btn = st.button("▶  Train & Optimize", use_container_width=True)

    st.markdown("---")
    st.markdown(
        f"<div style='font-family:IBM Plex Mono;font-size:0.65rem;color:#8b949e'>"
        f"Sheet: <b>{sheet_name}</b><br>"
        f"Inputs: {', '.join(input_cols_all)}<br>"
        f"Targets: {', '.join(output_cols_all)}</div>",
        unsafe_allow_html=True
    )


"""─────────────────────────────────────────────────────────────────────────"""
"""  MAIN HEADER                                                            """
"""─────────────────────────────────────────────────────────────────────────"""

st.markdown(
    "<h1 style='margin-bottom:0;font-size:1.7rem'>R-744 Hybrid System — Surrogate Modeling Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<span class='section-tag'>Active: {sheet_name} › {target_col} › {model_name}</span>",
    unsafe_allow_html=True
)
st.markdown("---")


"""─────────────────────────────────────────────────────────────────────────"""
"""  DATA PREVIEW TAB AREA                                                  """
"""─────────────────────────────────────────────────────────────────────────"""

with st.expander("📋 Data Preview", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Rows", f"{len(df_sheet):,}")
    with col2:
        metric_card("Input Features", str(len(input_cols_all)))
    with col3:
        metric_card("Target Vars", str(len(output_cols_all)))
    st.dataframe(
        df_sheet[input_cols_all + output_cols_all].dropna().head(20),
        use_container_width=True, height=220
    )


"""─────────────────────────────────────────────────────────────────────────"""
"""  TRAINING LOGIC (triggered by button)                                  """
"""─────────────────────────────────────────────────────────────────────────"""

if run_btn:
    with st.spinner("Preprocessing data…"):
        df_json = df_sheet.to_json()
        X_tr, X_te, y_tr, y_te, sx, sy, feat_names = preprocess(
            df_json, tuple(input_cols_all), target_col
        )

    default_est, param_grid = build_estimator(model_name)

    with st.spinner("Training default model…"):
        fitted_default = train_model(default_est, X_tr, y_tr)
        m_before = evaluate_model(fitted_default, X_te, y_te, sy)

    with st.spinner(f"Running RandomizedSearchCV (n_iter={N_ITER})…"):
        default_est2, param_grid2 = build_estimator(model_name)
        try:
            best_est, best_params = optimize_model(default_est2, param_grid2, X_tr, y_tr)
            m_after = evaluate_model(best_est, X_te, y_te, sy)
        except Exception as e:
            st.warning(f"Optimization failed ({e}). Using default model.")
            best_est    = fitted_default
            best_params = {}
            m_after     = m_before

    """ Persist in session state """
    st.session_state.m_before     = m_before
    st.session_state.m_after      = m_after
    st.session_state.best_model   = best_est
    st.session_state.scaler_X     = sx
    st.session_state.scaler_y     = sy
    st.session_state.best_params  = best_params
    st.session_state.input_cols   = input_cols_all
    st.session_state.feature_names = feat_names

    st.success("Training complete!")


"""─────────────────────────────────────────────────────────────────────────"""
"""  RESULTS SECTION                                                        """
"""─────────────────────────────────────────────────────────────────────────"""

m_before = st.session_state.m_before
m_after  = st.session_state.m_after

if m_before and m_after:

    st.markdown("### 📊 Model Results")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            "<div class='section-tag'>Before Optimization</div>",
            unsafe_allow_html=True
        )
        r1, r2, r3 = st.columns(3)
        with r1: metric_card("RMSE", f"{m_before['RMSE']:.4f}")
        with r2: metric_card("MSE",  f"{m_before['MSE']:.4f}")
        with r3: metric_card("R²",   f"{m_before['R2']:.4f}",
                             r2_class(m_before["R2"]))

    with c2:
        st.markdown(
            "<div class='section-tag'>After Optimization</div>",
            unsafe_allow_html=True
        )
        r1, r2, r3 = st.columns(3)
        with r1: metric_card("RMSE", f"{m_after['RMSE']:.4f}")
        with r2: metric_card("MSE",  f"{m_after['MSE']:.4f}")
        with r3: metric_card("R²",   f"{m_after['R2']:.4f}",
                             r2_class(m_after["R2"]))

    """ Best hyperparameters """
    if st.session_state.best_params:
        with st.expander("🔧 Best Hyperparameters", expanded=False):
            st.json(st.session_state.best_params)

    """ RMSE improvement callout """
    delta_rmse = m_before["RMSE"] - m_after["RMSE"]
    delta_r2   = m_after["R2"]    - m_before["R2"]
    col_a, col_b = st.columns(2)
    with col_a:
        arrow = "▼" if delta_rmse >= 0 else "▲"
        color = ACCENT_G if delta_rmse >= 0 else ACCENT_R
        st.markdown(
            f"<div style='text-align:center;font-family:IBM Plex Mono;font-size:0.85rem;"
            f"color:{color}'>{arrow} RMSE improved by {abs(delta_rmse):.4f}</div>",
            unsafe_allow_html=True
        )
    with col_b:
        arrow = "▲" if delta_r2 >= 0 else "▼"
        color = ACCENT_G if delta_r2 >= 0 else ACCENT_R
        st.markdown(
            f"<div style='text-align:center;font-family:IBM Plex Mono;font-size:0.85rem;"
            f"color:{color}'>{arrow} R² improved by {abs(delta_r2):.4f}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")


"""─────────────────────────────────────────────────────────────────────────"""
"""  REAL-TIME PREDICTION PANEL                                             """
"""─────────────────────────────────────────────────────────────────────────"""

st.markdown("### 🎯 Real-Time Prediction")

if st.session_state.best_model is None:
    st.info("Train a model first using the sidebar controls.")
else:
    input_cols_active = st.session_state.input_cols
    sx = st.session_state.scaler_X
    sy = st.session_state.scaler_y
    best_model = st.session_state.best_model

    """ Dynamic input sliders — use data range for sensible bounds """
    df_clean = df_sheet[input_cols_active + [target_col]].dropna()
    user_inputs = []

    n_inputs = len(input_cols_active)
    cols_per_row = min(n_inputs, 3)
    rows_needed  = (n_inputs + cols_per_row - 1) // cols_per_row
    flat_cols    = []

    for row_i in range(rows_needed):
        row_cols = st.columns(cols_per_row)
        flat_cols.extend(row_cols)

    for i, feat in enumerate(input_cols_active):
        import math

col_data = df_clean[feat].dropna()

# Fallback if column is empty
if col_data.empty:
    mn, mx, med = 0.0, 1.0, 0.5
else:
    mn  = float(col_data.min())
    mx  = float(col_data.max())
    med = float(col_data.median())

# Fix NaN / Inf
if not np.isfinite(mn): mn = 0.0
if not np.isfinite(mx): mx = mn + 1.0
if not np.isfinite(med): med = (mn + mx) / 2

# Ensure valid range
if mn == mx:
    mx = mn + 1e-6

# Clamp median into range
med = max(mn, min(med, mx))

# Safe step
step = (mx - mn) / 200
if not np.isfinite(step) or step <= 0:
    step = max(abs(mx - mn) * 0.01, 1e-6)

# Final slider (wrapped in try for safety)
try:
    val = flat_cols[i].slider(
        feat,
        min_value=float(mn),
        max_value=float(mx),
        value=float(med),
        step=float(step),
        format="%.3f"
    )
except Exception as e:
    st.warning(f"⚠️ Slider fallback used for '{feat}' ({e})")
    val = flat_cols[i].number_input(
        feat,
        value=float(med),
        step=float(step),
        format="%.3f"
    )

user_inputs.append(val)

    """ Predict instantly """
    try:
        X_user = sx.transform(np.array(user_inputs).reshape(1, -1))
        y_pred_s = best_model.predict(X_user)
        y_pred_val = float(
            sy.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()[0]
        )

        st.markdown(
            f"""<div class='pred-box'>
              <div class='pred-label'>{target_col} — Predicted Value</div>
              <div class='pred-value'>{y_pred_val:.4f}</div>
            </div>""",
            unsafe_allow_html=True
        )

        """ Log & download button """
        col_log, col_dl = st.columns([3, 1])
        with col_log:
            if st.button("💾 Log This Prediction"):
                log_prediction(
                    sheet_name, model_name, target_col,
                    user_inputs, input_cols_active, y_pred_val,
                    m_before, m_after
                )
                st.success("Prediction logged.")

        with col_dl:
            if os.path.exists(LOG_FILE):
                log_bytes = open(LOG_FILE, "rb").read()
                st.download_button(
                    "⬇ Download Log",
                    data=log_bytes,
                    file_name="prediction_log.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")


"""─────────────────────────────────────────────────────────────────────────"""
"""  GRAPH DASHBOARD                                                        """
"""─────────────────────────────────────────────────────────────────────────"""

st.markdown("### 📈 Visual Dashboard")

if m_before and m_after:
    tabs = st.tabs([
        "Before vs After",
        "Error Distribution",
        "Feature Importance",
        "Model Comparison",
    ])

    with tabs[0]:
        fig_ba = plot_before_after(m_before, m_after)
        st.pyplot(fig_ba, use_container_width=True)
        plt.close(fig_ba)

    with tabs[1]:
        fig_err = plot_error_distribution(m_after["errors"])
        st.pyplot(fig_err, use_container_width=True)
        plt.close(fig_err)

    with tabs[2]:
        feat_names, importances = get_feature_importance(
            st.session_state.best_model,
            model_name,
            st.session_state.feature_names or input_cols_all
        )
        if feat_names is not None:
            fig_fi = plot_feature_importance(feat_names, importances, model_name)
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)
        else:
            st.info(f"Feature importance is not available for {model_name}.")

    with tabs[3]:
        if os.path.exists(LOG_FILE):
            log_df = pd.read_csv(LOG_FILE)
            fig_mc = plot_model_comparison(log_df, target_col)
            if fig_mc:
                st.pyplot(fig_mc, use_container_width=True)
                plt.close(fig_mc)
            else:
                st.info("Train and log at least 2 different models to see comparison.")
        else:
            st.info("No prediction log yet. Log some predictions to enable comparison.")

else:
    st.info("Run training from the sidebar to populate the dashboard.")


"""─────────────────────────────────────────────────────────────────────────"""
"""  MODEL DOWNLOAD                                                         """
"""─────────────────────────────────────────────────────────────────────────"""

st.markdown("---")
st.markdown("### 📦 Download Trained Model")

if st.session_state.best_model is not None:
    col_dl1, col_dl2 = st.columns([2, 2])
    with col_dl1:
        model_payload = {
            "model":        st.session_state.best_model,
            "scaler_X":     st.session_state.scaler_X,
            "scaler_y":     st.session_state.scaler_y,
            "input_cols":   st.session_state.input_cols,
            "target":       target_col,
            "sheet":        sheet_name,
            "model_name":   model_name,
        }
        buf = io.BytesIO()
        joblib.dump(model_payload, buf)
        buf.seek(0)
        st.download_button(
            "⬇ Download Model (.pkl)",
            data=buf,
            file_name=f"model_{sheet_name}_{target_col}_{model_name.replace(' ', '_')}.pkl",
            mime="application/octet-stream",
        )

    with col_dl2:
        """ P_gc highlight """
        if target_col == PRIMARY_TARGET and m_after:
            st.markdown(
                f"<div style='background:#1a2f1a;border:1px solid #3fb950;"
                f"border-radius:8px;padding:10px 14px;"
                f"font-family:IBM Plex Mono;font-size:0.8rem;color:#3fb950'>"
                f"⭐ This is a P_gc model<br>"
                f"R² = {m_after['R2']:.4f} | RMSE = {m_after['RMSE']:.4f}"
                f"</div>",
                unsafe_allow_html=True
            )
else:
    st.info("Train a model first to enable download.")


"""─────────────────────────────────────────────────────────────────────────"""
"""  EXPERIMENT LOG TABLE                                                   """
"""─────────────────────────────────────────────────────────────────────────"""

st.markdown("---")
st.markdown("### 🗂 Experiment Log")

if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
    """ Show most recent 30 entries, newest first """
    display_df = log_df.tail(30).iloc[::-1].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, height=260)

    """ Best P_gc model highlight """
    pgc_log = log_df[log_df["Target"] == PRIMARY_TARGET]
    if not pgc_log.empty:
        best_row = pgc_log.loc[pgc_log["R2_after"].idxmax()]
        st.markdown(
            f"<div style='font-family:IBM Plex Mono;font-size:0.72rem;"
            f"color:#3fb950;margin-top:6px'>"
            f"⭐ Best logged P_gc model: <b>{best_row['Model']}</b> "
            f"on sheet <b>{best_row['Sheet']}</b> — "
            f"R² = {best_row['R2_after']:.4f}, RMSE = {best_row['RMSE_after']:.4f}"
            f"</div>",
            unsafe_allow_html=True
        )
else:
    st.caption("No predictions logged yet. Use the prediction panel to start logging.")


"""─────────────────────────────────────────────────────────────────────────"""
"""  FOOTER                                                                 """
"""─────────────────────────────────────────────────────────────────────────"""

st.markdown("---")
st.markdown(
    "<div style='font-family:IBM Plex Mono;font-size:0.62rem;color:#484f58;"
    "text-align:center'>R-744 Hybrid System Surrogate Modeler · "
    "Streamlit · scikit-learn · XGBoost · LightGBM · CatBoost</div>",
    unsafe_allow_html=True
)
