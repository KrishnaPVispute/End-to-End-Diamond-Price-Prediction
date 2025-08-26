import streamlit as st
import pandas as pd
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Diamond Price Prediction", layout="centered")

st.title("💎 Diamond Price Prediction")

st.markdown("Enter the diamond details below and click **Predict Price** to see the estimated value.")

# Input fields in main page (not sidebar)
carat = st.number_input("Carat", min_value=0.0, step=0.01, format="%.2f")
depth = st.number_input("Depth", min_value=0.0, step=0.1, format="%.1f")
table = st.number_input("Table", min_value=0.0, step=0.1, format="%.1f")
x = st.number_input("X (length in mm)", min_value=0.0, step=0.01, format="%.2f")
y = st.number_input("Y (width in mm)", min_value=0.0, step=0.01, format="%.2f")
z = st.number_input("Z (depth in mm)", min_value=0.0, step=0.01, format="%.2f")

cut = st.selectbox("Cut", options=["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", options=["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", options=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

# Predict button
if st.button(" Predict Price"):
    try:
        custom_data = CustomData(
            carat=carat,
            depth=depth,
            table=table,
            x=x,
            y=y,
            z=z,
            cut=cut,
            color=color,
            clarity=clarity
        )

        df = custom_data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)

        st.success(f" Predicted Diamond Price: **{prediction[0]:,.2f}**")

    except Exception as e:
        st.error(f" Error: {e}")




#streamlit run app.py
