import os
import streamlit as st
import pandas as pd
import joblib

# absolute safe path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)

st.title("📈 Sales Prediction App")

tv = st.number_input("TV Advertising Budget", min_value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0)

if st.button("Predict Sales"):
    user_input = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=["TV", "Radio", "Newspaper"]
    )

    prediction = model.predict(user_input)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")
