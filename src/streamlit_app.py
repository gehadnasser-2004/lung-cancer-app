import streamlit as st
import pickle
import numpy as np
import os 

# ----------------------------
# تحميل الموديل
# ----------------------------
model_path = os.path.join(os.path.dirname(__file__), "..", "lung_cancer_model.pkl")
model = pickle.load(open(model_path, "rb"))

# ----------------------------
# واجهة التطبيق
# ----------------------------
st.title("🩺 Lung Cancer Prediction App")
st.write("This app predicts the risk of lung cancer based on user inputs.")

# ----------------------------
# إدخالات المستخدم (نفس ترتيب الـ features)
# ----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
smoking = st.selectbox("Smoking", ["Yes", "No"])
yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
anxiety = st.selectbox("Anxiety", ["Yes", "No"])
peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
fatigue = st.selectbox("Fatigue", ["Yes", "No"])
allergy = st.selectbox("Allergy", ["Yes", "No"])
wheezing = st.selectbox("Wheezing", ["Yes", "No"])
alcohol = st.selectbox("Alcohol Consuming", ["Yes", "No"])
coughing = st.selectbox("Coughing", ["Yes", "No"])
shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

# ----------------------------
# تحويل المدخلات لقيم رقمية (0/1)
# ----------------------------
def encode_yes_no(value):
    return 1 if value == "Yes" else 0

gender_val = 1 if gender == "Male" else 0
smoking_val = encode_yes_no(smoking)
yellow_fingers_val = encode_yes_no(yellow_fingers)
anxiety_val = encode_yes_no(anxiety)
peer_pressure_val = encode_yes_no(peer_pressure)
chronic_disease_val = encode_yes_no(chronic_disease)
fatigue_val = encode_yes_no(fatigue)
allergy_val = encode_yes_no(allergy)
wheezing_val = encode_yes_no(wheezing)
alcohol_val = encode_yes_no(alcohol)
coughing_val = encode_yes_no(coughing)
shortness_of_breath_val = encode_yes_no(shortness_of_breath)
swallowing_difficulty_val = encode_yes_no(swallowing_difficulty)
chest_pain_val = encode_yes_no(chest_pain)

# ----------------------------
# زر التنبؤ
# ----------------------------
if st.button("Predict"):
    # ترتيب الأعمدة لازم يكون نفس ترتيب التدريب
    input_data = np.array([[gender_val, age, smoking_val, yellow_fingers_val,
                            anxiety_val, peer_pressure_val, chronic_disease_val,
                            fatigue_val, allergy_val, wheezing_val, alcohol_val,
                            coughing_val, shortness_of_breath_val, swallowing_difficulty_val,
                            chest_pain_val]])
    
    prediction = model.predict(input_data)

    # عرض النتيجة
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Lung Cancer")
    else:
        st.success("✅ Low Risk of Lung Cancer")
