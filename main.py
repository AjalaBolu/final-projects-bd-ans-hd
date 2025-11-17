import streamlit as st 
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from streamlit_option_menu import option_menu

working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_models = pickle.load(open(f'{working_dir}/models/AdvancedDB.pkl', 'rb'))
diabetes_scaler = pickle.load(open(f'{working_dir}/models/adb_scaler.pkl', 'rb'))

NewBMI_Overweight = 0
NewBMI_Underweight = 0
NewBMI_obesity_1 = 0
NewBMI_obesity_2 = 0
NewBMI_obesity_3 = 0
NewInsulinScore_Normal = 0
NewGlucose_low = 0
NewGlucose_Normal = 0
NewGlucose_Overweight = 0
NewGlucose_Secret = 0

def get_cleaned_data():
    data = pd.read_csv("datasets/diabetis.csv") 
    return data

def add_sidebar():
    st.sidebar.header("Patients Data")
    data = get_cleaned_data()

    slider_labels = [
        ("Number of pregnancies", "Pregnancies"),
        ("Glucose level", "Glucose"),
        ("Blood Pressure level", "BloodPressure"),
        ("Skin Thickness level", "SkinThickness"),
        ("Insulin level", "Insulin"),
        ("BMI level", "BMI"),
        ("Diabetes Pedigree Function level", "DiabetesPedigreeFunction"),
        ("Age of the person", "Age")
    ]

    user_data = {}

    for label, key in slider_labels:
        col_min = data[key].min()
        col_max = data[key].max()
        col_mean = data[key].mean()

        # BMI and Pedigree Function can be floats, others are integers
        if key in ["BMI", "DiabetesPedigreeFunction"]:
            user_data[key] = st.sidebar.slider(
                label,
                float(col_min),
                float(col_max),
                float(col_mean)
            )
        else:
            user_data[key] = st.sidebar.slider(
                label,
                int(col_min),
                int(col_max),
                int(col_mean)
            )

    return user_data

def get_scaled_values(user_data):
    data = get_cleaned_data()

    X = data.drop(['Outcome'], axis=1)

    scaled_data = {}

    for key, value in user_data.items():
        col_min = X[key].min()
        col_max = X[key].max()
        scaled_value = (value - col_min) / (col_max - col_min)
        scaled_data[key] = scaled_value

    return scaled_data

def get_radar_chart(user_data):
    user_data = get_scaled_values(user_data)

    fig = go.Figure(data=go.Scatterpolar(
        r=[
            user_data['Pregnancies'],
                    user_data['Glucose'],
                    user_data['BloodPressure'],
                    user_data['SkinThickness'],
                    user_data['Insulin'],
                    user_data['BMI'],
                    user_data['DiabetesPedigreeFunction'],
                    user_data['Age']
        ],
        theta=['Pregnancies','Glucose',
               'BloodPressure','SkinThickness',
                'Insulin','BMI',
                'DiabetesPedigreeFunction','Age'],
        fill='toself'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0,1]
            ),
        ),
        showlegend=False
     )
    return fig

def predict_diabetes(user_data, selected_model_name):
    input_values = [
        user_data['Pregnancies'],
        user_data['Glucose'],
        user_data['BloodPressure'],
        user_data['SkinThickness'],
        user_data['Insulin'],
        user_data['BMI'],
        user_data['DiabetesPedigreeFunction'],
        user_data['Age'],
    ]
    extra_features = [
        NewBMI_Overweight,
        NewBMI_Underweight,
        NewBMI_obesity_1,
        NewBMI_obesity_2,
        NewBMI_obesity_3,
        NewInsulinScore_Normal,
        NewGlucose_low,
        NewGlucose_Normal,
        NewGlucose_Overweight,
        NewGlucose_Secret,
    ]
    X = np.array(input_values + extra_features).reshape(1, -1)

    # ---- scale and pick model ----
    X_scaled = diabetes_scaler.transform(X)
    model = diabetes_models[selected_model_name]

    # ---- predict class ----
    y_pred = model.predict(X_scaled)
    is_diabetic = int(y_pred[0]) == 1

    # ---- get probabilities safely ----
    prob_0, prob_1 = None, None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]   # [p(class0), p(class1)]
            prob_0, prob_1 = float(proba[0]), float(proba[1])
        elif hasattr(model, "decision_function"):
            # convert margin to pseudo-probability with sigmoid
            margin = float(model.decision_function(X_scaled)[0])
            prob_1 = 1.0 / (1.0 + np.exp(-margin))
            prob_0 = 1.0 - prob_1
        else:
            # fallback: no proba available
            prob_1 = float(is_diabetic)
            prob_0 = 1.0 - prob_1
    except Exception:
        # absolute fallback if anything goes wrong above
        prob_1 = float(is_diabetic)
        prob_0 = 1.0 - prob_1

    # ---- display nicely ----
    if is_diabetic:
        st.error("📈 This model predicts you're **likely Diabetic**")
        st.write(f"🔴 Probability of being Diabetic: **{prob_1 * 100:.2f}%**")
        st.write(f"🟢 Probability of being Healthy: **{prob_0 * 100:.2f}%**")
    else:
        st.success("📉 This model predicts you're **likely Healthy**")
        st.write(f"🟢 Probability of being Healthy: **{prob_0 * 100:.2f}%**")
        st.write(f"🔴 Probability of being Diabetic: **{prob_1 * 100:.2f}%**")
    
    st.write("This app can assist in preliminary health assessments but is not a substitute for professional medical advice. Always consult a healthcare provider for accurate diagnosis and treatment.")
    
def main():
    st.set_page_config(
        page_title="Advanced Diabetes Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    user_data = add_sidebar()

    with st.container():
        st.title("Diabetes Predictor 🩺")
        st.write("This is a web application that predicts whether a person has diabetes or not based on their health parameters.") 
        selected_model_name = st.selectbox("Select a model", list(diabetes_models.keys()))

    col1, col2 = st.columns([3,2])

    with col1:
        st.subheader("Patient Health Parameters")
        radar_chart = get_radar_chart(user_data)
        st.plotly_chart(radar_chart)
    with col2:
        st.subheader("Diabetes Prediction")
        predict_diabetes(user_data, selected_model_name)

if __name__ == '__main__':
    main()
