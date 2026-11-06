
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("bike_demand_model_compressed.joblib")

st.set_page_config(
    page_title="Bike Rental Demand Predictor",
    page_icon="🚲",
    layout="centered"
)

st.title("Bike Rental Demand Predictor")
st.write(
    "Enter time and weather conditions to estimate "
    "hourly bike rental demand."
)

st.subheader("Time Conditions")

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox(
        "Year",
        [0, 1],
        index=1,
        format_func=lambda x: "2011" if x == 0 else "2012"
    )

    month = st.slider("Month", 1, 12, 6)
    day = st.slider("Day", 1, 31, 15)
    hour = st.slider("Hour", 0, 23, 17)

with col2:
    weekday = st.selectbox(
        "Weekday (0 = Sunday)",
        list(range(7)),
        index=2
    )

    workingday = st.selectbox(
        "Working day?",
        [0, 1],
        index=1,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    holiday = st.selectbox(
        "Holiday?",
        [0, 1],
        index=0,
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.subheader("Weather Conditions")

col3, col4 = st.columns(2)

with col3:
    season = st.selectbox(
        "Season",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Spring",
            2: "Summer",
            3: "Fall",
            4: "Winter"
        }[x]
    )

    weather = st.selectbox(
        "Weather",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Clear",
            2: "Mist",
            3: "Light rain/snow",
            4: "Heavy rain/snow"
        }[x]
    )

with col4:
    temperature = st.slider(
        "Temperature (normalized)",
        0.0, 1.0, 0.5, 0.01
    )

    feels_like = st.slider(
        "Feels-like temperature (normalized)",
        0.0, 1.0, 0.5, 0.01
    )

    humidity = st.slider(
        "Humidity (normalized)",
        0.0, 1.0, 0.5, 0.01
    )

    windspeed = st.slider(
        "Windspeed (normalized)",
        0.0, 1.0, 0.2, 0.01
    )

if st.button("Predict Bike Demand", type="primary"):
    values = {
        "yr": int(year),
        "mnth": int(month),
        "hr": int(hour),
        "holiday": int(holiday),
        "weekday": int(weekday),
        "workingday": int(workingday),
        "temp": float(temperature),
        "atemp": float(feels_like),
        "hum": float(humidity),
        "windspeed": float(windspeed),
        "day": int(day),
        "dayofweek": int(weekday),
    }

    for i in range(1, 5):
        values[f"season_{i}"] = int(season == i)
        values[f"weathersit_{i}"] = int(weather == i)

    input_df = pd.DataFrame(
        [[values[col] for col in model.feature_names_in_]],
        columns=model.feature_names_in_
    )

    prediction = max(
        0,
        round(float(model.predict(input_df)[0]))
    )

    st.success(f"Estimated bike rentals: {prediction}")
    st.caption("Prediction represents estimated hourly bike rentals.")
