import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
try:
    model = joblib.load("random_forest_model.joblib")
except FileNotFoundError:
    st.error("Model file not found. Please upload the model.")
    st.stop()

# Streamlit App Title
st.title("Car Price Prediction App 🚗")

# Input Fields
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=300000, value=20000)
owner = st.selectbox("Owner Type", [0, 1, 2, 3])
present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, max_value=50.0, value=5.0)
original_fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
original_seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
original_transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Prediction Button
if st.button("Predict Price"):
    # Convert categorical inputs to numerical values
    fuel_mapping = {"Petrol": 0, "Diesel": 1}
    seller_mapping = {"Dealer": 0, "Individual": 1}
    transmission_mapping = {"Manual": 0, "Automatic": 1}

    fuel_type_encoded = fuel_mapping[original_fuel_type]
    seller_type_encoded = seller_mapping[original_seller_type]
    transmission_encoded = transmission_mapping[original_transmission]

    # One-hot encode
    fuel_petrol = 1 if fuel_type_encoded == 0 else 0
    fuel_diesel = 1 if fuel_type_encoded == 1 else 0
    seller_dealer = 1 if seller_type_encoded == 0 else 0
    seller_individual = 1 if seller_type_encoded == 1 else 0
    transmission_manual = 1 if transmission_encoded == 0 else 0
    transmission_automatic = 1 if transmission_encoded == 1 else 0

    # Create a feature DataFrame with exactly 15 features
    user_data = pd.DataFrame([[year, present_price, kms_driven, owner, fuel_petrol, fuel_diesel, 
                               seller_dealer, seller_individual, transmission_manual, transmission_automatic, 
                               2024 - year, 1.5, 20, 5, 0]],  # Placeholder values
                             columns=['year', 'present_price', 'kms_driven', 'owner', 'fuel_Petrol', 'fuel_Diesel', 
                                      'seller_Dealer', 'seller_Individual', 'transmission_Manual', 'transmission_Automatic', 
                                      'car_age', 'engine_size', 'mileage', 'seats', 'is_4wd'])

    try:
        # Predicting the car price
        prediction = model.predict(user_data)
        
        # If prediction is a 2D array or multi-element array, extract the scalar value
        if isinstance(prediction, np.ndarray):
            predicted_price = prediction.flatten()[0]  # Flatten the array and extract the first element
        
        # Display the result
        st.success(f"Predicted Car Price: ₹{predicted_price:,.2f} Lakhs")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
