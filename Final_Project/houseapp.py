import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="AI Real Estate SaaS",
    page_icon="🏡",
    layout="wide"
)

# ===============================
# SAFE PATH BASE
# ===============================
BASE_DIR = os.path.dirname(__file__)

# ===============================
# SAFE LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "house_price_model.pkl")
    
    if not os.path.exists(model_path):
        st.error("❌ Model file not found!")
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ===============================
# SAFE LOAD DATA
# ===============================
@st.cache_data
def load_data():
    data_path = os.path.join(BASE_DIR, "housing.csv")  # FIXED (was data.csv)

    if not os.path.exists(data_path):
        st.error("❌ Data file not found!")
        st.stop()

    return pd.read_csv(data_path)

df = load_data()

# ===============================
# HEADER
# ===============================
st.markdown("""
# 🏡 AI Real Estate Intelligence Platform  
### Modern SaaS Dashboard for Property Price Prediction
""")

st.success("🚀 Live AI System Running")

st.divider()

# ===============================
# KPI METRICS
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Dataset", len(df))
col2.metric("🏠 Avg Price", f"${df['price'].mean():,.0f}")
col3.metric("📏 Avg Size", f"{df['sqft_living'].mean():,.0f} sqft")
col4.metric("🧠 Model", "AI ML")

st.divider()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🔧 Predict House Price")

sqft = st.sidebar.slider("Select House Size", 300, 10000, 1500)
predict = st.sidebar.button("🚀 Predict Now")

# ===============================
# MAIN LOGIC
# ===============================
if predict:

    prediction = model.predict(np.array([[sqft]]))[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("📏 Size", f"{sqft} sqft")
    c2.metric("💰 Price", f"${prediction:,.0f}")
    c3.metric("📈 Status", "Predicted")

    st.divider()

    # ===============================
    # GRAPH SAFETY CHECK
    # ===============================
    if "sqft_living" in df.columns and "price" in df.columns:

        st.subheader("📊 Market Overview")
        fig, ax = plt.subplots()
        ax.scatter(df["sqft_living"], df["price"], alpha=0.4)
        ax.set_xlabel("House Size")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        st.subheader("🎯 Your Property Position")
        fig, ax = plt.subplots()
        ax.scatter(df["sqft_living"], df["price"], alpha=0.3)
        ax.scatter([sqft], [prediction], color="red", s=120)
        st.pyplot(fig)

        st.subheader("📉 Price Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["price"], bins=30)
        st.pyplot(fig)

    # ===============================
    # DOWNLOAD RESULT
    # ===============================
    result_df = pd.DataFrame({
        "Sqft": [sqft],
        "Predicted Price": [prediction]
    })

    st.download_button(
        "⬇️ Download Prediction",
        result_df.to_csv(index=False),
        "prediction.csv",
        "text/csv"
    )

    # ===============================
    # INSIGHTS
    # ===============================
    st.info("""
    📌 Insights:
    - Larger homes → higher prices  
    - Model captures linear trend  
    - Real estate AI prediction active  
    """)

else:
    st.info("👈 Use sidebar to generate AI prediction")
