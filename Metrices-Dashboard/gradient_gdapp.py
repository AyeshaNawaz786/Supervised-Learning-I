import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# PAGE CONFIG (SaaS LOOK)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Regression Lab SaaS",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────────
# STYLE (DARK SaaS DASHBOARD)
# ─────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #0d1117; }
.stApp { background-color: #0d1117; }

.big-title {
    font-size: 34px;
    font-weight: 800;
    background: linear-gradient(90deg,#58a6ff,#d2a8ff,#3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────
st.markdown('<div class="big-title">📊 AI Regression Metrics SaaS Lab</div>', unsafe_allow_html=True)
st.write("Industrial-level ML dashboard with Metrics + Gradient Descent")

# ─────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")

noise = st.sidebar.slider("Noise Level", 0, 100, 25)
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.05)
iters = st.sidebar.slider("Iterations", 50, 500, 200)

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
np.random.seed(42)

X, y = make_regression(n_samples=300, n_features=5, noise=noise, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)

# ─────────────────────────────────────────────
# DASHBOARD METRICS
# ─────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R² Score", f"{r2:.3f}")

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
st.markdown("## 📈 Model Performance")

fig = plt.figure(figsize=(10,5))
plt.scatter(y_test, pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
st.pyplot(fig)

# ─────────────────────────────────────────────
# GRADIENT DESCENT (SIMPLE SaaS VERSION)
# ─────────────────────────────────────────────
st.markdown("## ⚡ Gradient Descent Simulation")

Xg = np.linspace(0, 10, 100)
yg = 3.5 * Xg + 7 + np.random.normal(0, 8, 100)

x = (Xg - Xg.mean()) / Xg.std()
y2 = (yg - yg.mean()) / yg.std()

w = 0
b = 0
costs = []

for _ in range(iters):
    yhat = w * x + b
    err = yhat - y2
    w -= lr * np.mean(2 * err * x)
    b -= lr * np.mean(2 * err)
    costs.append(np.mean(err**2))

fig2 = plt.figure()
plt.plot(costs, color="cyan")
plt.title("Cost Function (Gradient Descent)")
st.pyplot(fig2)

# ─────────────────────────────────────────────
# DOWNLOAD REPORT
# ─────────────────────────────────────────────
st.markdown("## 📥 Export")

report = f"""
MAE: {mae}
MSE: {mse}
RMSE: {rmse}
R2: {r2}
"""

st.download_button(
    "Download Metrics Report",
    report,
    file_name="ml_report.txt"
)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("🚀 Built as SaaS-level ML Dashboard | Streamlit + Sklearn + NumPy")