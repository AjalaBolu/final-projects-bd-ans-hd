import streamlit as st
import pickle
import os
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Final Year Project", layout="wide")

working_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------- Load Models ----------------
# Heart disease models
heart_models = pickle.load(open(f"{working_dir}/models/Advancedheart.pkl", "rb"))
heart_scaler = pickle.load(open(f"{working_dir}/models/adheart_scaler.pkl", "rb"))

# Fake news detection models
fake_models = pickle.load(open(f"{working_dir}/models/FakeNewsModels.pkl", "rb"))
fake_vectorizer = pickle.load(open(f"{working_dir}/models/fakenews_vectorizer.pkl", "rb"))


# ---------------- Sidebar ----------------
with st.sidebar:
    selected = option_menu(
        "Final Year Projects",
        ['Heart Disease Prediction', 'Fake News Detection'],
        menu_icon='hospital-fill',
        icons=['heart', 'newspaper'],
        default_index=0
    )


# ---------------- HEART DISEASE PAGE ----------------
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
        restecg = st.text_input("Resting ECG results")
    with col2:
        thalach = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        exang = st.text_input("Exercise Induced Angina")
    with col1:
        oldpeak = st.text_input("ST Depression")
    with col2:
        slope = st.text_input("Slope of ST segment")
    with col3:
        ca = st.text_input("Major vessels colored by flourosopy")
    with col1:
        thal = st.text_input("Thal (0 = normal, 1 = fixed defect, 2 = reversible)")

    if st.button("Heart Disease Test Result"):
        if not all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
            st.error("⚠️ Please fill in all fields before testing.")
        else:
            try:
                user_input = [float(x) for x in [
                    age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal
                ]]

                user_input = np.array([user_input])
                user_input_scaled = heart_scaler.transform(user_input)

                model = heart_models[selected_model_name]
                prediction = model.predict(user_input_scaled)
                probability = model.predict_proba(user_input_scaled)[0][1]

                if prediction[0] == 1:
                    st.error(f"📈 Elevated risk detected (Confidence: {probability*100:.2f}%)")
                else:
                    st.success(f"📉 Reduced risk (Confidence: {(1-probability)*100:.2f}%)")

            except ValueError:
                st.error("⚠️ Invalid input. Please enter numeric values only.")



# ---------------- FAKE NEWS DETECTION PAGE ----------------
if selected == 'Fake News Detection':
    st.title("Fake News Detection Using Machine Learning 📰🔍")

    selected_fake_model = st.selectbox("Select a model", list(fake_models.keys()))

    news_text = st.text_area("Enter the news text below:")

    if st.button("Analyze News"):
        if news_text.strip() == "":
            st.error("⚠️ Please enter news text.")
        else:
            # Convert text → TF-IDF features
            vectorized_text = fake_vectorizer.transform([news_text])

            # Predict
            model = fake_models[selected_fake_model]
            prediction = model.predict(vectorized_text)[0]

            # If model supports probability
            try:
                prob = model.predict_proba(vectorized_text)[0][1]
            except:
                prob = None

            # Display results
            if prediction == 1:
                if prob:
                    st.error(f"🚨 Fake News Detected (Confidence: {prob*100:.2f}%)")
                else:
                    st.error("🚨 Fake News Detected!")
            else:
                if prob:
                    st.success(f"✅ Real / Legitimate News (Confidence: {(1-prob)*100:.2f}%)")
                else:
                    st.success("✅ Real / Legitimate News")
