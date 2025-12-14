import streamlit as st
import pickle
import numpy as np

# Load model
with open("decision_tree_model.pkl", "rb") as f:
    model, le_branch, le_course, le_grade = pickle.load(f)

st.title("🎓 Student Grade Prediction App")

# Inputs
branch = st.selectbox("Select Branch", le_branch.classes_)
course = st.selectbox("Select Course", le_course.classes_)
marks = st.slider("Enter Marks", 0, 100, 75)

# Encode inputs
branch_encoded = le_branch.transform([branch])[0]
course_encoded = le_course.transform([course])[0]

# Prediction
if st.button("Predict Grade"):
    input_data = np.array([[branch_encoded, course_encoded, marks]])
    pred = model.predict(input_data)
    grade = le_grade.inverse_transform(pred)
    st.success(f"Predicted Grade: {grade[0]}")
