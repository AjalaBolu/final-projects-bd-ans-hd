import streamlit as st
import pickle
import os
import numpy as np
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Final Year Project", layout="wide")

working_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------- Load Models ----------------
# Load dictionaries of models (Scikit-learn)

heart_models = pickle.load(open(f'{working_dir}/models/Advancedheart.pkl', 'rb'))

# Load scalers (Pickle)

heart_scaler = pickle.load(open(f'{working_dir}/models/adheart_scaler.pkl', 'rb'))


# Load deep learning model (Keras)


# ---------------- Sidebar Menu ----------------
with st.sidebar:
    selected = option_menu(
        "Final Year Projects",
        ['Heart Disease Prediction'],
        menu_icon='hospital-fill',
        icons=['heart'],
        default_index=0
    )

# ---------------- Heart Disease Prediction ----------------
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction Using Machine Learning ❤️")

    selected_model_name = st.selectbox("Select a model", list(heart_models.keys()))

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age")
    with col2:
        sex = st.text_input("Sex")
    with col3:
        cp = st.text_input("Chest Pain Types")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Serum cholestoral in mg/dl")
    with col3:
        fbs = st.text_input("Fasting blood sugar > 120 mg/dl")
    with col1:
        restecg = st.text_input("Resting Electrocardiographic results")
    with col2:
        thalach = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        exang = st.text_input("Exercise Induced Angina")
    with col1:
        oldpeak = st.text_input("ST depression induced by exercise")
    with col2:
        slope = st.text_input("Slope of the peak exercise ST segment")
    with col3:
        ca = st.text_input("Major vessels colored by flourosopy")
    with col1:
        thal = st.text_input("Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect")

    if st.button("Heart Disease Test Result"):
        if not all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
            st.error("⚠️ Please fill in all fields before testing.")
        else:
            try:
                user_input = [float(x) for x in [
                    age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal
                ]]

                # Make input 2D
                user_input = np.array([user_input])

                # Scale input
                user_input_scaled = heart_scaler.transform(user_input)

                # Predict
                model = heart_models[selected_model_name]
                prediction = model.predict(user_input_scaled)
                probability = model.predict_proba(user_input_scaled)[0][1]

                if prediction[0] == 1:
                    st.error(f"📈 Prediction: Elevated heart disease risk detected (Confidence: {probability*100:.2f}%)")
                else:
                    st.success(f"📉 Prediction: Reduced risk of heart disease (Confidence: {(1-probability)*100:.2f}%)")

            except ValueError:
                st.error("⚠️ Invalid input. Please enter numeric values only.")

