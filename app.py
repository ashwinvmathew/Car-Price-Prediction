import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("car_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Car Price Prediction 🚗💰")
st.write("Enter car details to predict the selling price.")

# User input fields

present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, format="%.2f")
kms_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
car_age = st.number_input("Car Age (in years)", min_value=0)
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])

# Encode categorical values
seller_type = 1 if seller_type == "Individual" else 0
transmission = 1 if transmission == "Automatic" else 0
fuel_type_cng = 1 if fuel_type == "CNG" else 0
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0

# Scale 'Kms_Driven'
kms_driven_scaled = scaler.transform(np.array([[kms_driven]]))[0][0]

# Prepare input for prediction
features = np.array([[present_price, kms_driven_scaled, owner, seller_type, transmission, car_age, fuel_type_cng, fuel_type_diesel, fuel_type_petrol]])

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Selling Price: ₹{prediction:.2f} Lakhs")
