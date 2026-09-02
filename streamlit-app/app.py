import streamlit as st
import pickle
import os
import numpy as np
import tensorflow as tf
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

# Breast cancer model (Keras DNN, 30 input features -> 2 classes: benign/malignant)
breast_model = tf.keras.models.load_model(f"{working_dir}/models/AdvancedbreastDnn.keras")
# NOTE: this DNN was almost certainly trained on scaled features (e.g. StandardScaler
# on the sklearn Wisconsin Breast Cancer dataset). If you saved a scaler during
# training, pickle-load it here the same way as heart_scaler and use it below.
# breast_scaler = pickle.load(open(f"{working_dir}/models/breast_scaler.pkl", "rb"))


# ---------------- Sidebar ----------------
with st.sidebar:
    selected = option_menu(
        "Final Year Projects",
        ['Heart Disease Prediction', 'Fake News Detection', 'Breast Cancer Prediction'],
        menu_icon='hospital-fill',
        icons=['heart', 'newspaper', 'gender-female'],
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


# ---------------- BREAST CANCER PAGE ----------------
if selected == 'Breast Cancer Prediction':
    st.title("Breast Cancer Prediction Using Deep Learning 🎗️")

    st.caption("Enter the tumor's measured cell nuclei features below (values from the "
                "Wisconsin Breast Cancer diagnostic dataset).")

    # The 30 features from the standard Wisconsin Breast Cancer dataset,
    # grouped as mean / standard error ("se") / worst, 3 per row.
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
    ]

    cols = st.columns(3)
    breast_inputs = {}
    for i, feature in enumerate(feature_names):
        with cols[i % 3]:
            label = feature.replace("_", " ").title()
            breast_inputs[feature] = st.text_input(label, key=f"breast_{feature}")

    if st.button("Breast Cancer Test Result"):
        values = list(breast_inputs.values())
        if not all(values):
            st.error("⚠️ Please fill in all fields before testing.")
        else:
            try:
                user_input = np.array([[float(v) for v in values]])

                # If you have a fitted scaler for this model, apply it here instead:
                # user_input = breast_scaler.transform(user_input)

                prediction = breast_model.predict(user_input)
                # Output is 2-class softmax: index 0 = benign, index 1 = malignant
                # (double-check this matches your training label encoding)
                probability = prediction[0][1]
                predicted_class = int(np.argmax(prediction[0]))

                if predicted_class == 1:
                    st.error(f"📈 Malignant (Confidence: {probability*100:.2f}%)")
                else:
                    st.success(f"📉 Benign (Confidence: {(1-probability)*100:.2f}%)")

            except ValueError:
                st.error("⚠️ Invalid input. Please enter numeric values only.")