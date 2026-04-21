import streamlit as st

st.set_page_config(page_title="R-744 Modeler", layout="wide")

st.title("R-744 Surrogate Modeler")

# ===================== SIDEBAR =====================
with st.sidebar:
    uploaded = st.file_uploader("Upload Excel", type=["xlsx", "xls"])

    if uploaded is None:
        st.stop()

    sheets_dict = load_data(uploaded.read())
    sheet_name = st.selectbox("Sheet", list(sheets_dict.keys()))
    df_sheet = sheets_dict[sheet_name]

    input_cols_all, output_cols_all = detect_columns(df_sheet)

    if not input_cols_all or not output_cols_all:
        st.error("Missing required columns")
        st.stop()

    target_col = st.selectbox("Target", output_cols_all)
    model_name = st.selectbox("Model", MODEL_NAMES)

    run_btn = st.button("Train Model")

# ===================== DATA =====================
st.subheader("Data Preview")
st.dataframe(df_sheet.head())

# ===================== TRAIN =====================
if run_btn:
    with st.spinner("Training..."):
        df_json = df_sheet.to_json()

        X_tr, X_te, y_tr, y_te, sx, sy, feat_names = preprocess(
            df_json, tuple(input_cols_all), target_col
        )

        est, grid = build_estimator(model_name)

        model = train_model(est, X_tr, y_tr)
        m_before = evaluate_model(model, X_te, y_te, sy)

        best_model, best_params = optimize_model(est, grid, X_tr, y_tr)
        m_after = evaluate_model(best_model, X_te, y_te, sy)

        st.session_state.update({
            "model": best_model,
            "sx": sx,
            "sy": sy,
            "m_before": m_before,
            "m_after": m_after,
            "features": input_cols_all
        })

    st.success("Training complete")

# ===================== RESULTS =====================
if "m_after" in st.session_state:
    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Before Optimization")
        st.write(st.session_state.m_before)

    with col2:
        st.write("After Optimization")
        st.write(st.session_state.m_after)

# ===================== PREDICTION =====================
if "model" in st.session_state:
    st.subheader("Prediction")

    inputs = []
    for feat in st.session_state.features:
        val = st.number_input(feat, value=0.0)
        inputs.append(val)

    if st.button("Predict"):
        X_user = st.session_state.sx.transform([inputs])
        y_pred = st.session_state.model.predict(X_user)
        y_val = st.session_state.sy.inverse_transform(y_pred.reshape(-1,1))[0][0]

        st.success(f"Prediction: {y_val:.4f}")

# ===================== PLOTS =====================
if "m_after" in st.session_state:
    st.subheader("Plots")

    fig1 = plot_before_after(
        st.session_state.m_before,
        st.session_state.m_after
    )
    st.pyplot(fig1)

    fig2 = plot_error_distribution(
        st.session_state.m_after["errors"]
    )
    st.pyplot(fig2)

# ===================== DOWNLOAD =====================
if "model" in st.session_state:
    import io, joblib

    buf = io.BytesIO()
    joblib.dump(st.session_state.model, buf)

    st.download_button("Download Model", buf.getvalue(), "model.pkl")  
