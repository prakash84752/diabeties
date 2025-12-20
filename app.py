import streamlit as st
import pickle
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model file path (must match filename exactly)
MODEL_PATH = os.path.join(BASE_DIR, "diabeties.pkl")

# Load the model safely
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please check diabeties.pkl is in the same folder as app.py.")
    st.stop()

st.title("Diabetes Prediction App")

# Example input
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    result = model.predict([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    if result[0] == 1:
        st.success("The person is diabetic")
    else:
        st.success("The person is not diabetic")


