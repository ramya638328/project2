import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

st.title("🎓 Student Grade Prediction")

# Load dataset
df = pd.read_csv("StudentMarksDataset.csv")

def Grade_class(marks):
    if marks >= 80:
        return "A"
    elif marks >= 70:
        return "B"
    else:
        return "C"

df["Grade"] = df["Std_Marks"].apply(Grade_class)

le_branch = LabelEncoder()
le_course = LabelEncoder()
le_grade = LabelEncoder()

df["Std_Branch"] = le_branch.fit_transform(df["Std_Branch"])
df["Std_Course"] = le_course.fit_transform(df["Std_Course"])
df["Grade"] = le_grade.fit_transform(df["Grade"])

X = df[["Std_Branch", "Std_Course", "Std_Marks"]]
y = df["Grade"]

model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
model.fit(X, y)

branch = st.selectbox("Select Branch", le_branch.classes_)
course = st.selectbox("Select Course", le_course.classes_)
marks = st.slider("Enter Marks", 0, 100, 75)

branch_encoded = le_branch.transform([branch])[0]
course_encoded = le_course.transform([course])[0]

if st.button("Predict Grade"):
    pred = model.predict([[branch_encoded, course_encoded, marks]])
    grade = le_grade.inverse_transform(pred)
    st.success(f"Predicted Grade: {grade[0]}")
