import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    layout="wide",
    page_icon="📊"
)

# ---------------- HEADER ----------------
st.title("📊 Sales Prediction Dashboard")
st.markdown("🚀 ML Project | Advertising Budget → Sales Prediction")

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

# =========================================================
# 🟢 SINGLE PREDICTION SECTION
# =========================================================
st.sidebar.header("📌 Single Prediction Input")

tv = st.sidebar.slider("TV Budget", 0, 500, 100)
radio = st.sidebar.slider("Radio Budget", 0, 500, 50)
newspaper = st.sidebar.slider("Newspaper Budget", 0, 500, 30)

st.markdown("---")
st.subheader("📌 Single Prediction Mode")

approve_single = st.button("✅ Approve & Predict")

if approve_single:

    input_df = pd.DataFrame([[tv, radio, newspaper]],
                             columns=["TV", "Radio", "Newspaper"])

    pred = model.predict(input_df)
    pred_value = pred[0]

    st.success(f"📈 Predicted Sales: {pred_value:.2f}")

    # Business logic
    if pred_value < 10:
        label = "Weak Investment"
        st.error("📉 Low Sales Expected")
    elif pred_value < 20:
        label = "Average ROI"
        st.warning("📊 Medium Sales Expected")
    else:
        label = "Strong Marketing Performance"
        st.success("📈 High Sales Expected")

    st.info(f"🧾 Status: {label}")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_value,
        title={'text': "Sales Meter"},
        gauge={
            'axis': {'range': [0, 30]},
            'steps': [
                {'range': [0, 10], 'color': "red"},
                {'range': [10, 20], 'color': "yellow"},
                {'range': [20, 30], 'color': "green"}
            ]
        }
    ))

    st.plotly_chart(fig)

# =========================================================
# 🟢 DATASET INSIGHTS (OPTIONAL)
# =========================================================
st.markdown("---")
st.subheader("📉 Dataset Insights")

try:
    df = pd.read_csv("dataset.csv")

    fig2, ax = plt.subplots()
    ax.scatter(df["TV"], df["Sales"], label="TV vs Sales")
    ax.scatter(df["Radio"], df["Sales"], label="Radio vs Sales")

    ax.set_xlabel("Advertising Budget")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig2)

except:
    st.warning("⚠ Dataset not found for visualization")

# =========================================================
# 🟢 BULK PREDICTION (UPLOAD + APPROVE + DOWNLOAD)
# =========================================================
st.markdown("---")
st.subheader("📂 Bulk Prediction Mode")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain TV, Radio, Newspaper)",
    type=["csv"]
)

approve_bulk = st.button("📂 Approve & Run Bulk Prediction")

if uploaded_file is not None:

    df_upload = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Preview")
    st.dataframe(df_upload)

    required_cols = ["TV", "Radio", "Newspaper"]

    if approve_bulk:

        if all(col in df_upload.columns for col in required_cols):

            df_upload = df_upload[required_cols]

            predictions = model.predict(df_upload)

            df_upload["Predicted Sales"] = predictions

            st.success("✅ Bulk Prediction Completed")

            st.dataframe(df_upload)

            # DOWNLOAD BUTTON
            csv = df_upload.to_csv(index=False).encode('utf-8')

            st.download_button(
                "⬇ Download Excel Report",
                csv,
                "sales_prediction_report.csv",
                "text/csv"
            )

        else:
            st.error("❌ CSV must contain TV, Radio, Newspaper columns")

# =========================================================
# 🟢 FOOTER
# =========================================================
st.markdown("---")
st.markdown("🚀 Built by Nithiya Sri G| AIDS RMKCET 🔥| Sales Prediction System")
