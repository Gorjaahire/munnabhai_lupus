import streamlit as st
import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("hospital_dataset.csv")

# Load trained model
with open("bed_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("AI-Powered Hospital Bed & Resource Predictor")
st.write("This system uses machine learning to forecast hospital bed usage for the next 48 hours.")

# --- Prepare last data row ---
last_row = df[["ER_patients", "Scheduled_surgeries", "current_beds_full", "ICU_full", "staff_on_duty"]].tail(1)

# --- AI prediction ---
predicted_beds = model.predict(last_row)[0]

st.subheader("🔮 AI Prediction for Next Hour")
st.write(f"**Predicted bed usage:** {predicted_beds:.2f}")

# Convert to resources
predicted_staff = 15 + predicted_beds * 0.2

st.write(f"**Required staff:** {predicted_staff:.0f}")

# Chart
st.subheader("📈 Past 48 Hours — Bed Usage")
st.line_chart(df["current_beds_full"].tail(48))
st.subheader("📊 Data Overview")
st.dataframe(df.tail(10))
