import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# CONFIG (STARTUP STYLE)
# ===============================
st.set_page_config(
    page_title="AI Real Estate SaaS",
    page_icon="🏡",
    layout="wide"
)

# ===============================
# LOAD MODEL + DATA
# ===============================
model = pickle.load(open("house_price_model.pkl", "rb"))
df = pd.read_csv("data.csv")

# ===============================
# LANDING HEADER (SAAS STYLE)
# ===============================
st.markdown("""
# 🏡 AI Real Estate Intelligence Platform  
### Modern SaaS Dashboard for Property Price Prediction
""")

st.success("🚀 Live AI System Project Ready")

st.divider()

# ===============================
# KPI CARDS (UBER/AIRBNB STYLE)
# ===============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Dataset", len(df))
col2.metric("🏠 Avg Price", f"${df['price'].mean():,.0f}")
col3.metric("📏 Avg Size", f"{df['sqft_living'].mean():,.0f} sqft")
col4.metric("🧠 Model", "AI ML")

st.divider()

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.title("🔧 Predict House Price")

sqft = st.sidebar.slider("Select House Size", 300, 10000, 1500)
predict = st.sidebar.button("🚀 Predict Now")

# ===============================
# MAIN LOGIC
# ===============================
if predict:

    prediction = model.predict(np.array([[sqft]]))[0]

    # ===========================
    # RESULT CARDS
    # ===========================
    c1, c2, c3 = st.columns(3)

    c1.metric("📏 Size", f"{sqft} sqft")
    c2.metric("💰 Price", f"${prediction:,.0f}")
    c3.metric("📈 Status", "Predicted")

    st.divider()

    # ===========================
    # GRAPH 1: MARKET VIEW
    # ===========================
    st.subheader("📊 Market Overview")

    fig, ax = plt.subplots()
    ax.scatter(df["sqft_living"], df["price"], alpha=0.4)
    ax.set_xlabel("House Size")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # ===========================
    # GRAPH 2: YOUR POINT
    # ===========================
    st.subheader("🎯 Your Property Position")

    fig, ax = plt.subplots()
    ax.scatter(df["sqft_living"], df["price"], alpha=0.3)
    ax.scatter([sqft], [prediction], color="red", s=120)
    st.pyplot(fig)

    # ===========================
    # GRAPH 3: PRICE DISTRIBUTION
    # ===========================
    st.subheader("📉 Price Distribution")

    fig, ax = plt.subplots()
    ax.hist(df["price"], bins=30)
    st.pyplot(fig)

    # ===========================
    # INSIGHT BOX
    # ===========================
    st.info("""
    📌 Insights:
    - Larger homes → higher prices  
    - Data shows linear market trend  
    - AI model captures real estate patterns  
    """)

else:
    st.info("👈 Use sidebar to generate AI prediction")