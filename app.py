# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
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
            './model/nairobi_uv_model.pkl',
            'app/nairobi_uv_model.pkl',
            './app/nairobi_uv_model.pkl'
        ]
        
        model_path = None
        scaler_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                # Try to find scaler in same location
                scaler_candidates = [
                    path.replace('nairobi_uv_model.pkl', 'scaler.pkl'),
                    path.replace('nairobi_uv_model.pkl', 'scaler.pkl').replace('./', ''),
                    'scaler.pkl',
                    './scaler.pkl',
                    'na/scaler.pkl',
                    './na/scaler.pkl'
                ]
                for scaler_candidate in scaler_candidates:
                    if os.path.exists(scaler_candidate):
                        scaler_path = scaler_candidate
                        break
                if scaler_path:
                    break
        
        if model_path is None:
            st.error("Model file not found. Please ensure 'nairobi_uv_model.pkl' is accessible.")
            return None, None
        if scaler_path is None:
            st.error("Scaler file not found. Please ensure 'scaler.pkl' is accessible.")
            return None, None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        st.success(f"✅ Model loaded successfully!")
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
        - Ensure both `nairobi_uv_model.pkl` and `scaler.pkl` are in your project
        - Common locations: same folder, 'na/' folder, or 'model/' folder
        - Check the file names are exactly correct
        """)
        
        # Show current directory contents
        st.subheader("📁 Current Directory Contents")
        try:
            current_files = []
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if file.endswith('.pkl'):
                        current_files.append(os.path.join(root, file))
            
            if current_files:
                st.write("Found .pkl files:")
                for file in current_files:
                    st.write(f"- {file}")
            else:
                st.write("No .pkl files found in current directory")
        except Exception as e:
            st.write(f"Could not list directory contents: {e}")
        
        return
    
    # Sidebar for user input
    st.sidebar.header("📊 Input Weather Parameters")
    st.sidebar.markdown("Enter the weather conditions to predict UV index risk:")
    
    # User input fields with session state for persistence
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
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        temp = st.slider("Temperature (°F)", min_value=50.0, max_value=95.0, 
                        value=st.session_state.temp, step=0.5, key="temp_slider")
        humidity = st.slider("Humidity (%)", min_value=30.0, max_value=95.0, 
                           value=st.session_state.humidity, step=1.0, key="humidity_slider")
        precip = st.slider("Precipitation (inches)", min_value=0.0, max_value=2.0, 
                          value=st.session_state.precip, step=0.1, key="precip_slider")
    
    with col2:
        solar_radiation = st.slider("Solar Radiation (W/m²)", min_value=100.0, max_value=350.0, 
                                  value=st.session_state.solar_radiation, step=5.0, key="solar_slider")
        cloud_cover = st.slider("Cloud Cover (%)", min_value=0.0, max_value=100.0, 
                              value=st.session_state.cloud_cover, step=5.0, key="cloud_slider")
        sun_duration = st.slider("Sun Duration (minutes)", min_value=600.0, max_value=780.0, 
                               value=st.session_state.sun_duration, step=5.0, key="sun_slider")
    
    # Create input array
    input_data = np.array([[temp, humidity, precip, solar_radiation, cloud_cover, sun_duration]])
    
    # Prediction button
    if st.sidebar.button("🔍 Predict UV Risk", use_container_width=True):
        # Update session state
        st.session_state.update({
            'temp': temp,
            'humidity': humidity,
            'precip': precip,
            'solar_radiation': solar_radiation,
            'cloud_cover': cloud_cover,
            'sun_duration': sun_duration
        })
        
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

    # Quick predictions section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Quick Scenarios")
    
    scenarios = {
        "☀️ Sunny Day": [80.0, 55.0, 0.0, 300.0, 15.0, 750.0],
        "☁️ Cloudy Day": [72.0, 75.0, 0.1, 160.0, 80.0, 700.0],
        "🌧️ Rainy Day": [68.0, 85.0, 0.8, 120.0, 95.0, 650.0]
    }
    
    for scenario_name, scenario_values in scenarios.items():
        if st.sidebar.button(scenario_name, use_container_width=True):
            st.session_state.temp = scenario_values[0]
            st.session_state.humidity = scenario_values[1]
            st.session_state.precip = scenario_values[2]
            st.session_state.solar_radiation = scenario_values[3]
            st.session_state.cloud_cover = scenario_values[4]
            st.session_state.sun_duration = scenario_values[5]
            st.rerun()

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