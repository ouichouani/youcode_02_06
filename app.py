
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Mental Health Risk Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Mental Health Risk Prediction")
st.write("Enter the student's information to predict their mental health risk.")

# Load trained objects
model = joblib.load("models/best_svm_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")
target_encoder = joblib.load("models/target_encoder.joblib")

st.header("Student Information")

with st.form("prediction_form"):

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Non-binary"]
    )

    year = st.selectbox(
        "Year of Study",
        [1, 2, 3, 4, 5]
    )

    sleep = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0
    )

    study = st.number_input(
        "Study Hours Per Week",
        min_value=0.0,
        value=20.0
    )

    gpa = st.number_input(
        "GPA",
        min_value=0.0,
        value=2.5
    )

    social = st.number_input(
        "Social Media Hours",
        min_value=0.0,
        value=3.0
    )

    screen = st.number_input(
        "Screen Time Hours",
        min_value=0.0,
        value=5.0
    )

    activity = st.selectbox(
        "Physical Activity",
        ["Low", "Moderate", "High"]
    )

    financial = st.number_input(
        "Financial Stress",
        min_value=0.0,
        value=5.0
    )

    family = st.number_input(
        "Family Support",
        min_value=0.0,
        value=5.0
    )

    attendance = st.number_input(
        "Attendance Rate",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    part_time = st.selectbox(
        "Part Time Job",
        ["Yes", "No"]
    )

    diet = st.selectbox(
        "Diet Quality",
        ["Poor", "Average", "Good"]
    )

    history = st.selectbox(
        "Previous Mental Health History",
        ["Yes", "No"]
    )

    submit = st.form_submit_button("Predict")

# Prediction
if submit:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Year_of_Study": year,
        "Sleep_Hours": sleep,
        "Study_Hours_Per_Week": study,
        "GPA": gpa,
        "Social_Media_Hours": social,
        "Screen_Time_Hours": screen,
        "Physical_Activity": activity,
        "Financial_Stress": financial,
        "Family_Support": family,
        "Attendance_Rate": attendance,
        "Part_Time_Job": part_time,
        "Diet_Quality": diet,
        "Previous_Mental_Health_History": history
    }])

    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(processed_data)

    # Decode label
    result = target_encoder.inverse_transform(prediction)

    st.success(f"Predicted Mental Health Risk: {result[0]}")
