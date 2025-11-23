import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import calendar

# Set page configuration
st.set_page_config(
    page_title="Nairobi UV Index Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    try:
        # Try different possible paths
        possible_paths = [
            'nairobi_uv_model.pkl',
            './nairobi_uv_model.pkl',
            'na/nairobi_uv_model.pkl',
            './na/nairobi_uv_model.pkl',
            'model/nairobi_uv_model.pkl',
            './model/nairobi_uv_model.pkl'
        ]
        
        model_path = None
        scaler_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                scaler_path = path.replace('nairobi_uv_model.pkl', 'scaler.pkl')
                if os.path.exists(scaler_path):
                    break
        
        if model_path is None or scaler_path is None:
            st.error("Model files not found. Please ensure both 'nairobi_uv_model.pkl' and 'scaler.pkl' are accessible.")
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        st.success(f"✅ Model loaded successfully from: {model_path}")
        return model, scaler
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def main():
    # Header
    st.title("☀️ Nairobi UV Index Prediction App")
    st.markdown("""
    This app predicts **high UV index days** (UV Index > 8) in Nairobi, Kenya using weather data.
    High UV index indicates increased risk of skin damage from sun exposure.
    """)
    
    # Load model
    with st.spinner("Loading prediction model..."):
        model, scaler = load_model()
    
    if model is None:
        st.info("💡 **Troubleshooting tips:**")
        st.markdown("""
        - Ensure both `nairobi_uv_model.pkl` and `scaler.pkl` are in your project folder
        - If they're in a subfolder, update the path in the code
        - Check that the files have read permissions
        """)
        return
    
    # Sidebar for user input
    st.sidebar.header("📊 Input Weather Parameters")
    st.sidebar.markdown("Enter the weather conditions to predict UV index risk:")
    
    # User input fields
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        temp = st.slider("Temperature (°F)", min_value=50.0, max_value=95.0, value=75.0, step=0.5)
        humidity = st.slider("Humidity (%)", min_value=30.0, max_value=95.0, value=70.0, step=1.0)
        precip = st.slider("Precipitation (inches)", min_value=0.0, max_value=2.0, value=0.1, step=0.1)
    
    with col2:
        solar_radiation = st.slider("Solar Radiation (W/m²)", min_value=100.0, max_value=350.0, value=200.0, step=5.0)
        cloud_cover = st.slider("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
        sun_duration = st.slider("Sun Duration (minutes)", min_value=600.0, max_value=780.0, value=720.0, step=5.0)
    
    # Create input array
    input_data = np.array([[temp, humidity, precip, solar_radiation, cloud_cover, sun_duration]])
    
    # Prediction button
    if st.sidebar.button("🔍 Predict UV Risk", use_container_width=True):
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display results
        st.header("📈 Prediction Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            if prediction == 1:
                st.error("🚨 High UV Risk Detected")
                st.metric("UV Index Category", "> 8 (Very High)", "High Risk")
                st.warning("""
                **Protection Required:**
                - Use sunscreen SPF 30+
                - Wear protective clothing
                - Seek shade 10AM-4PM
                - Wear UV-blocking sunglasses
                - Wear a wide-brimmed hat
                """)
            else:
                st.success("✅ Low UV Risk")
                st.metric("UV Index Category", "≤ 8 (Moderate/Low)", "Low Risk")
                st.info("""
                **Recommended:**
                - Basic sun protection
                - Enjoy outdoor activities
                - Stay hydrated
                """)
        
        with result_col2:
            st.subheader("Probability Analysis")
            st.metric("High UV Probability", f"{probability[1]:.1%}")
            st.metric("Low UV Probability", f"{probability[0]:.1%}")
            
            # Probability gauge
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh([0], [probability[1]], color='red' if probability[1] > 0.5 else 'green', alpha=0.6)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability of High UV')
            ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
            st.pyplot(fig)
        
        with result_col3:
            st.subheader("Weather Summary")
            weather_info = {
                "Temperature": f"{temp}°F",
                "Humidity": f"{humidity}%",
                "Precipitation": f"{precip} in",
                "Solar Radiation": f"{solar_radiation} W/m²",
                "Cloud Cover": f"{cloud_cover}%",
                "Sun Duration": f"{sun_duration} min"
            }
            
            for key, value in weather_info.items():
                st.write(f"**{key}:** {value}")
        
        # Feature importance
        st.subheader("🔍 Feature Importance Analysis")
        if hasattr(model, 'feature_importances_'):
            feature_names = ['Temperature', 'Humidity', 'Precipitation', 'Solar Radiation', 'Cloud Cover', 'Sun Duration']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                          color=plt.cm.viridis(importance_df['Importance']))
            ax.set_xlabel('Importance Score')
            ax.set_title('Which Factors Most Influence UV Prediction?')
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center')
            
            st.pyplot(fig)
    
    # Main content area
    st.markdown("---")
    
    # Educational sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Understanding UV Index")
        st.markdown("""
        **UV Index Scale:**
        - **1-2:** Low - No protection needed
        - **3-5:** Moderate - Protection recommended
        - **6-7:** High - Protection essential
        - **8-10:** Very High - Extra protection needed
        - **11+:** Extreme - Avoid sun exposure
        
        **Nairobi's UV Challenge:**
        - Equatorial location = higher UV exposure
        - Altitude: 1,795m = 15% more UV
        - Year-round high UV risk
        """)
    
    with col2:
        st.subheader("🌤️ Nairobi Weather Patterns")
        st.markdown("""
        **Typical Ranges:**
        - Temperature: 60-80°F year-round
        - Humidity: 60-80%
        - Solar Radiation: 150-300 W/m²
        - UV Index: Often 8-12
        
        **Seasonal Notes:**
        - **Dry seasons:** Higher UV risk
        - **Rainy seasons:** Temporary relief
        - **Consistent risk:** Due to equatorial location
        """)
    
    # Quick predictions section
    st.markdown("---")
    st.subheader("🎯 Try Common Weather Scenarios")
    
    scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
    
    # Pre-defined scenarios
    scenarios = {
        "Sunny Day": [80.0, 55.0, 0.0, 300.0, 15.0, 750.0],
        "Cloudy Day": [72.0, 75.0, 0.1, 160.0, 80.0, 700.0],
        "Rainy Day": [68.0, 85.0, 0.8, 120.0, 95.0, 650.0]
    }
    
    with scenario_col1:
        if st.button("☀️ Typical Sunny Day", use_container_width=True):
            st.session_state.temp = 80.0
            st.session_state.humidity = 55.0
            st.session_state.precip = 0.0
            st.session_state.solar_radiation = 300.0
            st.session_state.cloud_cover = 15.0
            st.session_state.sun_duration = 750.0
            st.rerun()
    
    with scenario_col2:
        if st.button("☁️ Overcast Day", use_container_width=True):
            st.session_state.temp = 72.0
            st.session_state.humidity = 75.0
            st.session_state.precip = 0.1
            st.session_state.solar_radiation = 160.0
            st.session_state.cloud_cover = 80.0
            st.session_state.sun_duration = 700.0
            st.rerun()
    
    with scenario_col3:
        if st.button("🌧️ Rainy Season Day", use_container_width=True):
            st.session_state.temp = 68.0
            st.session_state.humidity = 85.0
            st.session_state.precip = 0.8
            st.session_state.solar_radiation = 120.0
            st.session_state.cloud_cover = 95.0
            st.session_state.sun_duration = 650.0
            st.rerun()

    # Data insights section
    st.markdown("---")
    st.subheader("📈 Nairobi UV Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Monthly UV Risk Pattern**")
        # Sample visualization
        months = [calendar.month_name[i] for i in range(1, 13)]
        high_uv_prob = [0.7, 0.8, 0.9, 0.85, 0.8, 0.7, 0.6, 0.7, 0.8, 0.85, 0.8, 0.75]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(months, high_uv_prob, color=['red' if x > 0.7 else 'orange' for x in high_uv_prob])
        plt.xticks(rotation=45)
        plt.title('Probability of High UV Index by Month in Nairobi')
        plt.ylim(0, 1)
        plt.ylabel('Probability of High UV')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.0%}', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Protection Guidelines**")
        st.markdown("""
        **When UV > 8 (High Risk):**
        - Sunscreen: SPF 30+, reapply every 2 hours
        - Clothing: Long sleeves, UV-protective fabric
        - Timing: Avoid sun 10AM-4PM
        - Accessories: Hat, sunglasses required
        
        **General Tips for Nairobi:**
        - Check UV forecast daily
        - Even cloudy days can have high UV
        - Altitude increases UV exposure
        - Hydration helps with heat management
        """)

# Initialize session state for sliders
if 'temp' not in st.session_state:
    st.session_state.temp = 75.0
if 'humidity' not in st.session_state:
    st.session_state.humidity = 70.0
if 'precip' not in st.session_state:
    st.session_state.precip = 0.1
if 'solar_radiation' not in st.session_state:
    st.session_state.solar_radiation = 200.0
if 'cloud_cover' not in st.session_state:
    st.session_state.cloud_cover = 50.0
if 'sun_duration' not in st.session_state:
    st.session_state.sun_duration = 720.0

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Model: Random Forest Classifier | Data: Nairobi Weather Forecast</p>
    <p><small>Always consult local weather authorities for official UV index information. This tool provides predictions based on machine learning models.</small></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()