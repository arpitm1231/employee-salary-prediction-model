import streamlit as st
import pandas as pd
import joblib

# Load model (pipeline with preprocessing)
model = joblib.load("best_model.pkl")

# Streamlit page settings
st.set_page_config(page_title="Income Prediction App", layout="centered")
st.title("💼 Income Classification App")
st.write("Enter employee details to predict if income is `<=50K` or `>50K`.")

# Input form
def user_input():
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    workclass = st.number_input("Workclass (Encoded)", min_value=0, value=3)
    fnlwgt = st.number_input("FNLWGT", min_value=10000, value=200000)
    educational_num = st.number_input("Education Level (educational-num)", min_value=1, max_value=16, value=10)
    marital_status = st.number_input("Marital Status (marital-status)", min_value=0, value=1)
    occupation = st.number_input("Occupation (Encoded)", min_value=0, value=4)
    relationship = st.number_input("Relationship (Encoded)", min_value=0, value=1)
    race = st.number_input("Race (Encoded)", min_value=0, value=1)
    gender = st.number_input("Gender (0 = Female, 1 = Male)", min_value=0, max_value=1, value=1)
    capital_gain = st.number_input("Capital Gain", min_value=0, value=4)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=100, value=40)
    native_country = st.number_input("Native Country (Encoded)", min_value=0, value=38)

    data = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "educational-num": educational_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }
    return pd.DataFrame([data])

# Get user input
input_df = user_input()

# Predict on button click
if st.button("🔍 Predict Income"):
    # Optional: check what model expects and align
    try:
        input_df = input_df[model.feature_names_in_]  # Ensure column order matches
    except AttributeError:
        st.warning("⚠️ Warning: Model doesn't have `feature_names_in_` attribute. Proceeding without reordering.")

    # Debug: Show input columns (optional)
    st.write("📥 Input Preview:", input_df)

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted Income: **{prediction}**")

    if prediction == ">50K":
        st.markdown("💰 High income! 🎉")
    else:
        st.markdown("💼 Below or equal to 50K income.")
