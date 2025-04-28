
import streamlit as st
# This must be the first Streamlit command
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")

import numpy as np
import joblib
import plotly.express as px
import pandas as pd

# Cache the model loading to improve performance
@st.cache_resource
def load_models():
    """Load and cache the prediction models"""
    scaler = joblib.load('transformer_model.pkl')
    linreg = joblib.load('linear_model.pkl')
    return scaler, linreg

# Load the models
scaler, linreg = load_models()

# App header
st.title("🏡 House Price Predictor")
st.write("Enter house details below to estimate the price")

# Create two columns for a more compact layout
col1, col2 = st.columns(2)

with col1:
    square_footage = st.number_input("Square Footage", min_value=500, max_value=5000, value=1500)
    num_bedrooms = st.number_input("Bedrooms", min_value=1, max_value=5, value=3)
    num_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=3, value=2)
    year_built = st.number_input("Year Built", min_value=1950, max_value=2022, value=2000)

with col2:
    lot_size = st.number_input("Lot Size (acres)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    garage_size = st.radio("Garage Size (cars)", options=[0, 1, 2], index=1)
    neighborhood_quality = st.slider("Neighborhood Quality", 1, 10, 5, 
                                   help="1 = Poor, 10 = Excellent")

# Predict button
if st.button("Predict House Price", use_container_width=True):
    # Show a spinner during prediction
    with st.spinner("Calculating price..."):
        # Arrange input data
        input_data = np.array([[square_footage, num_bedrooms, num_bathrooms, year_built,
                               lot_size, garage_size, neighborhood_quality]])
        
        # Transform the input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        predicted_price = linreg.predict(input_scaled)[0]
    
    # Display the result with nice formatting
    st.success(f"Estimated House Price: ${predicted_price:,.2f}")
    
    # Create a simple feature importance chart (for demonstration)
    features = ['Square Footage', 'Bedrooms', 'Bathrooms', 'Year Built', 
               'Lot Size', 'Garage Size', 'Neighborhood']
    importance = [0.35, 0.15, 0.18, 0.12, 0.08, 0.05, 0.07]  # Example values
    
    fig = px.bar(
        x=importance, 
        y=features, 
        orientation='h',
        title="Feature Importance",
        labels={'x': 'Importance', 'y': 'Feature'},
        color=importance,
        color_continuous_scale='blues'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display example comparable houses
    st.subheader("Similar Properties")
    data = {
        'Square Footage': [square_footage-100, square_footage+200, square_footage-50],
        'Bedrooms': [num_bedrooms, num_bedrooms+1, num_bedrooms],
        'Bathrooms': [num_bathrooms-0.5, num_bathrooms, num_bathrooms+0.5],
        'Year Built': [year_built-5, year_built+10, year_built-15],
        'Price': [predicted_price*0.92, predicted_price*1.15, predicted_price*0.88]
    }
    df = pd.DataFrame(data)
    df['Price'] = df['Price'].map('${:,.2f}'.format)
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Made with Streamlit • Model last updated: January 2025")
