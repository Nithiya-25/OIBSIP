import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📊 Sales Prediction Dashboard")
st.markdown("🚀 Machine Learning Project | Streamlit Deployment")

# ---------------- LOAD MODEL SAFELY ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📌 Input Features")

tv = st.sidebar.slider("TV Advertising Budget", 0, 500, 100)
radio = st.sidebar.slider("Radio Advertising Budget", 0, 500, 50)
newspaper = st.sidebar.slider("Newspaper Advertising Budget", 0, 500, 30)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Prediction")

    if st.button("🚀 Predict Sales"):
        input_data = pd.DataFrame(
            [[tv, radio, newspaper]],
            columns=["TV", "Radio", "Newspaper"]
        )

        prediction = model.predict(input_data)

        st.success(f"📊 Predicted Sales: {prediction[0]:.2f}")

        st.info("💡 Insight: TV ads usually have highest impact on sales.")

with col2:
    st.subheader("📊 Input Summary")

    summary_df = pd.DataFrame({
        "Channel": ["TV", "Radio", "Newspaper"],
        "Budget": [tv, radio, newspaper]
    })

    st.bar_chart(summary_df.set_index("Channel"))

# ---------------- DATA VISUALIZATION ----------------
st.markdown("---")
st.subheader("📉 Dataset Visualization")

# If dataset exists
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
    st.warning("⚠ dataset.csv not found. Add dataset for visualization.")

# ---------------- INSIGHTS ----------------
st.markdown("---")
st.subheader("📌 Business Insights")

st.info("✔ TV advertising strongly influences sales growth")
st.info("✔ Balanced investment improves ROI")
st.info("✔ Data-driven decisions increase profit")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Developed by Nithiya Sri G| RMKCET")
