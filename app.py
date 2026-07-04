import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Telco Customer Churn Predictor")

# Load the saved model (trained in Malsrini.ipynb)
model = pickle.load(open("model.pkl", "rb"))

st.title("Telco Customer Churn Predictor")
st.write(
    "Enter a customer's details below to estimate the probability that "
    "they will churn (cancel their subscription)."
)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)

input_df = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
    "InternetService": internet_service, "OnlineSecurity": online_security,
    "OnlineBackup": online_backup, "DeviceProtection": device_protection,
    "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
    "Contract": contract, "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
}])

if st.button("Predict Churn Risk"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error(f"High churn risk — estimated probability: {prob:.1%}")
    else:
        st.success(f"Low churn risk — estimated probability: {prob:.1%}")

    st.progress(min(int(prob * 100), 100))

st.caption("Model: Random Forest classifier trained on the IBM Telco Customer Churn dataset. "
           "Built for COM763 Advanced Machine Learning — Portfolio Task 1.")
