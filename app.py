import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

@st.cache_resource
def load_trained_model():
    return joblib.load('best_student_model.pkl')

try:
    model = load_trained_model()
except Exception:
    st.error("Model file 'best_student_model.pkl' not found. Run train_model.py first.")
    st.stop()

st.title("Student Academic Performance Prediction")
st.markdown("Enter student academic metrics to predict the expected final exam score.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        study_hours = st.slider("Weekly Study Hours", 1.0, 50.0, 15.0, 0.5)
        attendance = st.slider("Attendance Percentage (%)", 0.0, 100.0, 80.0, 1.0)
        past_sgpa = st.number_input("Previous Semester SGPA (0 - 10)", 0.0, 10.0, 7.5, 0.1)
    with col2:
        internal_marks = st.slider("Internal Test Marks (out of 30)", 0.0, 30.0, 22.0, 0.5)
        assignments = st.slider("Assignments Completed (out of 10)", 0, 10, 8, 1)
        parent_edu = st.selectbox(
            "Parental Education Level",
            options=[0, 1, 2],
            format_func=lambda x: {0: "High School", 1: "Undergraduate / Bachelor", 2: "Postgraduate / Master"}[x]
        )
    submit = st.form_submit_button("Predict Final Score")

if submit:
    input_data = pd.DataFrame([{
        'Study_Hours': study_hours,
        'Attendance_Pct': attendance,
        'Past_SGPA': past_sgpa,
        'Internal_Marks': internal_marks,
        'Assignments_Completed': assignments,
        'Parent_Edu_Level': parent_edu
    }])
    
    pred_score = float(model.predict(input_data)[0])
    pred_score = max(0.0, min(100.0, pred_score))
    
    st.subheader(f"Predicted Final Marks: {pred_score:.1f} / 100")
    
    if pred_score >= 80:
        st.success("Performance Band: **Distinction / High Distinction**")
    elif pred_score >= 60:
        st.info("Performance Band: **First Division**")
    elif pred_score >= 45:
        st.warning("Performance Band: **Average / Passing**")
    else:
        st.error("Performance Band: **At-Risk / Academic Warning**")
        