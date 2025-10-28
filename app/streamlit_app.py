import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Page setup
# -----------------------------------------------------------
st.set_page_config(page_title="Sleep Quality Predictor", page_icon="😴", layout="centered")

st.title("🛌 Sleep Quality Predictor")
st.write("This AI model predicts whether your sleep quality is **Good** or **Poor** based on your daily habits.")

# -----------------------------------------------------------
# Load model safely
# -----------------------------------------------------------
model_path = "models/sleep_model.joblib"
if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please train the model first (run `scripts/train_model.py`).")
    st.stop()

model = joblib.load(model_path)

# -----------------------------------------------------------
# Sidebar info
# -----------------------------------------------------------
st.sidebar.header("ℹ️ About This Model")
st.sidebar.markdown("""
- **Model:** Random Forest Classifier  
- **Goal:** Predict sleep quality (Good / Poor)  
- **Trained on:** Synthetic lifestyle dataset  
- **Developer:** Sujal's AIML Project  
""")

if os.path.exists("results/feature_importances.csv"):
    fi_df = pd.read_csv("results/feature_importances.csv")
    st.sidebar.subheader("Top Feature Importances")
    st.sidebar.bar_chart(fi_df.set_index("feature").head(10))

# -----------------------------------------------------------
# User Input Form
# -----------------------------------------------------------
with st.form("sleep_form"):
    st.subheader("🧠 Enter Your Daily Details")
    col1, col2 = st.columns(2)

    with col1:
        hours_of_sleep = st.number_input("Hours of sleep", min_value=0.0, max_value=24.0, value=7.0)
        screen_time = st.number_input("Screen time (hours)", min_value=0.0, max_value=24.0, value=3.0)
        caffeine_intake = st.number_input("Cups of caffeinated drink", min_value=0, max_value=10, value=1)
        steps_walked = st.number_input("Steps walked", min_value=0, max_value=50000, value=6000)
        stress_level = st.slider("Stress level (1-10)", 1, 10, 4)

    with col2:
        exercise_minutes = st.number_input("Exercise minutes", min_value=0, max_value=500, value=20)
        alcohol_units = st.number_input("Alcohol units (0-10)", min_value=0, max_value=10, value=0)
        ambient_light = st.selectbox("Ambient light", ["low", "medium", "high"])
        weekday = st.selectbox("Day type", ["weekday", "weekend"])
        bedtime_hour = st.number_input("Bedtime hour (0-23)", min_value=0.0, max_value=23.99, value=23.0)

    submitted = st.form_submit_button("🔮 Predict Sleep Quality")

# -----------------------------------------------------------
# Prediction logic
# -----------------------------------------------------------
if submitted:
    try:
        X = pd.DataFrame([{
            "hours_of_sleep": hours_of_sleep,
            "screen_time": screen_time,
            "caffeine_intake": caffeine_intake,
            "steps_walked": steps_walked,
            "stress_level": stress_level,
            "exercise_minutes": exercise_minutes,
            "alcohol_units": alcohol_units,
            "ambient_light": ambient_light,
            "weekday": weekday,
            "bedtime_hour": bedtime_hour
        }])

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0, 1]  # probability of "good" class
        confidence = proba * 100

        st.subheader("🔎 Prediction Result")
        if pred == 1:
            st.success(f"🟢 **Good Sleep Quality!** ({confidence:.1f}% confidence)")
            st.balloons()
        else:
            st.error(f"🔴 **Poor Sleep Quality Detected** ({confidence:.1f}% confidence)")
            st.info("💡 Try improving: reduce screen time, caffeine, and stress. Sleep earlier!")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -----------------------------------------------------------
# Footer
# -----------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and RandomForest • Developed by Sujal (AIML Project)")
