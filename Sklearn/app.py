import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px

# ======================
# CONFIG
# ======================
st.set_page_config("Real Estate AI System", layout="wide", page_icon="🏠")

# ======================
# FILE SETUP
# ======================
USER_FILE = "users.csv"

if not os.path.exists(USER_FILE):
    pd.DataFrame(columns=["email", "password"]).to_csv(USER_FILE, index=False)

# ======================
# LOAD DATA + MODEL
# ======================
@st.cache_resource
def load_model():
    return pickle.load(open("best_model.pkl", "rb"))

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df.drop(["date","street","city","statezip","country"], axis=1, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

model = load_model()
df = load_data()
X = df.drop("price", axis=1)

# ======================
# SESSION STATE
# ======================
if "auth" not in st.session_state:
    st.session_state.auth = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ======================
# CLEAN NAME FUNCTION
# ======================
def get_name(email):
    return email.split("@")[0]

# ======================
# AUTH PAGE
# ======================
def auth_page():

    st.title("🏠 Real Estate AI System")

    option = st.radio("Choose Option", ["Login", "Register"], horizontal=True)

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    try:
        users = pd.read_csv(USER_FILE)
    except:
        users = pd.DataFrame(columns=["email", "password"])

    if "email" not in users.columns:
        users = pd.DataFrame(columns=["email", "password"])

    # REGISTER
    if option == "Register":

        if st.button("Create Account"):

            if email in users["email"].values:
                st.error("User already exists ❌")
            else:
                new_user = pd.DataFrame([[email, password]], columns=["email","password"])
                users = pd.concat([users, new_user], ignore_index=True)
                users.to_csv(USER_FILE, index=False)

                st.session_state.auth = True
                st.session_state.current_user = email

                st.success("Account created ✔")
                st.rerun()

    # LOGIN
    if option == "Login":

        if st.button("Login"):

            user = users[(users["email"] == email) & (users["password"] == password)]

            if not user.empty:
                st.session_state.auth = True
                st.session_state.current_user = email
                st.success("Login successful ✔")
                st.rerun()
            else:
                st.error("Invalid email or password ❌")

# ======================
# DASHBOARD
# ======================
def dashboard():

    display_name = get_name(st.session_state.current_user)

    st.sidebar.title(f"👤 {display_name}")

    menu = st.sidebar.radio("Navigation", [
        "Overview",
        "Prediction",
        "Analytics",
        "Logout"
    ])

    if menu == "Logout":
        st.session_state.auth = False
        st.session_state.current_user = None
        st.rerun()

    # ======================
    # OVERVIEW
    # ======================
    if menu == "Overview":

        st.title("📊 Market Overview")

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Listings", len(df))
        c2.metric("Average Price", f"${df['price'].mean():,.0f}")
        c3.metric("Max Price", f"${df['price'].max():,.0f}")

        st.dataframe(df.head(10), use_container_width=True)

    # ======================
    # PREDICTION
    # ======================
    elif menu == "Prediction":

        st.title("🏠 Price Prediction Engine")

        inputs = []

        colA, colB = st.columns(2)

        for i, col in enumerate(X.columns):

            with colA if i % 2 == 0 else colB:
                val = st.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
                inputs.append(val)

        if st.button("Predict Price"):

            pred = model.predict(np.array(inputs).reshape(1, -1))[0]

            st.success("Prediction Completed ✔")
            st.metric("Estimated Price", f"${pred:,.2f}")

    # ======================
    # ANALYTICS (UPGRADED)
    # ======================
    elif menu == "Analytics":

        st.title("📈 Advanced Analytics Dashboard")

        # 1. Histogram
        st.subheader("Price Distribution")
        fig1 = px.histogram(df, x="price", nbins=40)
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Box plot
        if "bedrooms" in df.columns:
            st.subheader("Bedrooms vs Price")
            fig2 = px.box(df, x="bedrooms", y="price")
            st.plotly_chart(fig2, use_container_width=True)

        # 3. Scatter
        if "sqft_living" in df.columns:
            st.subheader("Living Area vs Price")
            fig3 = px.scatter(df, x="sqft_living", y="price")
            st.plotly_chart(fig3, use_container_width=True)

        # 4. 🔥 CORRELATION HEATMAP (RESTORED)
        st.subheader("Feature Correlation Heatmap")

        corr = df.corr(numeric_only=True)

        fig4 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig4, use_container_width=True)

# ======================
# RUN APP
# ======================
if not st.session_state.auth:
    auth_page()
else:
    dashboard()