# ============================================================
#  STREAMLIT SAAS ML DASHBOARD (PRODUCTION LEVEL)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="ML SaaS", layout="wide")

st.title("🚀 AI Regression SaaS Dashboard")
st.markdown("Upload data → Get predictions instantly")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
st.success("✅ Model Loaded Successfully")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.header("⚙ Controls")

file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# ─────────────────────────────────────────────
# DATA INPUT
# ─────────────────────────────────────────────
if file is not None:
    data = pd.read_csv(file)
    st.subheader("📊 Input Data Preview")
    st.write(data.head())
else:
    st.info("No file uploaded → using demo data")

    data = pd.DataFrame(
        np.random.rand(10, 20),
        columns=[f"f{i}" for i in range(20)]
    )

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if st.button("🚀 Run Prediction"):

    try:
        predictions = model.predict(data)
        data["Prediction"] = predictions

        st.subheader("📈 Predictions")
        st.write(data.head())

        # Download CSV
        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇ Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

        st.success("Prediction Completed 🎯")

    except Exception as e:
        st.error(f"Error: {e}")

# ─────────────────────────────────────────────
# MODEL PERFORMANCE DEMO (OPTIONAL)
# ─────────────────────────────────────────────
st.sidebar.subheader("📊 Model Metrics (Demo)")

if st.sidebar.button("Show Metrics"):

    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    st.sidebar.write("MAE:", round(mae, 3))
    st.sidebar.write("MSE:", round(mse, 3))
    st.sidebar.write("RMSE:", round(rmse, 3))
    st.sidebar.write("R²:", round(r2, 3))

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("🔥 Built with Streamlit | ML SaaS Ready | Deployment Friendly")