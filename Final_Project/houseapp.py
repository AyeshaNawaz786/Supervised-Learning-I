import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="AI Real Estate SaaS", page_icon="🏡", layout="wide")

# ===============================
# LOAD MODEL + DATA
# ===============================
model = pickle.load(open("house_price_model.pkl", "rb"))
df = pd.read_csv("data.csv")

# Same preprocessing
df = df.drop(["date","street","city","statezip","country"], axis=1)
X = df.drop("price", axis=1)

# ===============================
# HEADER
# ===============================
st.markdown("""
<h1 style='text-align: center;'>🏡 AI Real Estate Intelligence Platform</h1>
<h4 style='text-align: center;'>Modern SaaS Dashboard</h4>
""", unsafe_allow_html=True)

st.success("🚀 Live AI System Ready")
st.divider()

# ===============================
# KPI
# ===============================
col1, col2, col3 = st.columns(3)
col1.metric("Dataset", len(df))
col2.metric("Avg Price", f"${df['price'].mean():,.0f}")
col3.metric("Features", len(X.columns))

st.divider()

# ===============================
# SIDEBAR INPUTS (AUTO GENERATE)
# ===============================
st.sidebar.title("🔧 Enter House Details")

input_data = []

for col in X.columns:
    val = st.sidebar.slider(
        col,
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    input_data.append(val)

predict = st.sidebar.button("🚀 Predict")

# ===============================
# PREDICTION
# ===============================
if predict:

    with st.spinner("🤖 AI is predicting..."):
        prediction = model.predict([input_data])[0]

    st.success(f"💰 Predicted Price: ${prediction:,.0f}")

    # Graph
    st.subheader("📊 Market Overview")

    fig, ax = plt.subplots()
    ax.scatter(df["sqft_living"], df["price"], alpha=0.4)
    st.pyplot(fig)

    # Download
    result_df = pd.DataFrame([input_data], columns=X.columns)
    result_df["Prediction"] = prediction

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Result",
        csv,
        "prediction.csv",
        "text/csv"
    )

else:
    st.info("👈 Enter values from sidebar and click Predict")
