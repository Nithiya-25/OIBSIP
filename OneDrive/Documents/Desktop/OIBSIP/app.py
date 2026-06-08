import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------- HEADER ----------------
st.title("📊 Sales Prediction Dashboard")
st.markdown("🚀 Machine Learning | OIBSIP Internship Project")

# ---------------- SAFE MODEL LOAD ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📌 Enter Advertising Budget")

tv = st.sidebar.slider("TV Budget", 0, 500, 100)
radio = st.sidebar.slider("Radio Budget", 0, 500, 50)
newspaper = st.sidebar.slider("Newspaper Budget", 0, 500, 30)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚀 Prediction")

    if st.button("Predict Sales"):
        input_df = pd.DataFrame(
            [[tv, radio, newspaper]],
            columns=["TV", "Radio", "Newspaper"]
        )

        prediction = model.predict(input_df)

        st.success(f"📈 Predicted Sales: {prediction[0]:.2f}")

        st.info("💡 Insight: TV ads usually give highest ROI")

with col2:
    st.subheader("📊 Budget Overview")

    chart_df = pd.DataFrame({
        "Channel": ["TV", "Radio", "Newspaper"],
        "Budget": [tv, radio, newspaper]
    })

    st.bar_chart(chart_df.set_index("Channel"))

# ---------------- DATA VISUALIZATION ----------------
st.markdown("---")
st.subheader("📉 Dataset Insights")

try:
    df = pd.read_csv("dataset.csv")

    fig, ax = plt.subplots()
    ax.scatter(df["TV"], df["Sales"], label="TV vs Sales")
    ax.scatter(df["Radio"], df["Sales"], label="Radio vs Sales")

    ax.set_xlabel("Advertising Budget")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

except:
    st.warning("Dataset not found for visualization")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Built by Nithiya Sri G | AI&DS RMKCET")
