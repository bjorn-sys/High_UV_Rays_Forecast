# High_UV_Rays_Forecast
---

**Nairobi UV Index Prediction & Weather Analysis**

A complete machine learning project designed to analyze one full year of Nairobi weather data (366 days) and predict days with high UV Index (UV > 8).
The project supports public health planning, outdoor activity safety, and early UV exposure warnings.

**📌 Project Overview**
---

This work focuses on identifying environmental patterns linked to UV radiation and building a predictive model that can classify days as safe UV or high UV. The project includes:

Exploratory data analysis (EDA)

Weather trend visualization

Feature engineering

Testing multiple machine-learning classifiers

Saving and preparing the best model for deployment

The final model is suitable for hospitals, weather agencies, schools, or mobile apps.



**📂 Dataset Summary***
---

366 rows, 33 weather features

Data includes temperatures, humidity, cloud cover, solar radiation, precipitation, wind, sunrise/sunset times, and UV index.

Weather data spans from May 2024 to May 2025.

A new target feature was created to indicate days with dangerously high UV levels.

**📊 EDA Highlights**
---

Temperature and solar radiation show strong seasonal patterns.

High UV days are associated with high temperature, low cloud cover, and increased solar radiation.

Several detailed visualizations were created, including:

UV distribution

Monthly weather seasonality

Solar radiation and temperature trends

Time-of-day sun duration patterns

Correlation analysis confirmed significant relationships with UV index.

**🤖 Machine Learning Summary**
---

The following models were trained and evaluated:

Logistic Regression

Random Forest

XGBoost

Decision Tree

Gradient Boosting

Best Model: Random Forest Classifier

Accuracy: 93%

Precision: 97%

Recall: 88%

Consistently outperformed other models

Exported for deployment alongside a trained feature scaler

**🧠 How the Prediction System Works**
---

The model uses several key environmental variables to determine whether a day will experience high UV.
The output is a simple two-class result:

0 → Safe UV Level

1 → High UV Level (Warning)

This makes it ideal for health advisories and automated alerts.

**🚀 Deployable App Options**
---

If deployed as a web or mobile application, the system can:

Provide daily UV risk assessments

Offer real-time alerts for high UV days

Visualize weather patterns for educational or planning purposes

The design is compatible with:

Streamlit

FastAPI

Flask

Dash

Mobile app backends

✨ Author

Emmanuel Bjorn
Machine Learning Engineer | Data Analyst
