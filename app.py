import streamlit as st
import pandas as pd
import random
import time

# --- Placeholder classes ---
class CustomData:
    def __init__(self, weather, road_condition, time_of_day, traffic, accident_type):
        self.weather = weather
        self.road_condition = road_condition
        self.time_of_day = time_of_day
        self.traffic = traffic
        self.accident_type = accident_type

    def get_data_as_data_frame(self):
        return pd.DataFrame({
            'weather': [self.weather],
            'road_condition': [self.road_condition],
            'time_of_day': [self.time_of_day],
            'traffic': [self.traffic],
            'accident_type': [self.accident_type]
        })

class PredictPipeline:
    def predict(self, features):
        severities = ["Low", "Medium", "High"]
        return [random.choice(severities)]
# --- End placeholder classes ---

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Accident Severity Predictor",
    page_icon="🚦",
    layout="centered",
)

# --- Custom CSS for enhanced 3D design with blinking title, highlights, and new features ---
st.markdown(
    """
    <style>
    /* Nighttime-inspired gradient background with animation */
    body {
        background: linear-gradient(135deg, #0a0f1c, #1a2e44, #2a4060);
        background-size: 400% 400%;
        animation: gradientAnimation 15s ease infinite;
        color: #e0e0ff; /* Light blue-white for contrast */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Centered container with 3D effect */
    .block-container {
        background: rgba(10, 15, 28, 0.8); /* Darker nighttime background */
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 5px 15px rgba(0, 100, 200, 0.3), 0 0 10px rgba(0, 150, 255, 0.2);
        border: 1px solid rgba(0, 150, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .block-container:hover {
        box-shadow: 0 8px 20px rgba(0, 100, 200, 0.5), 0 0 15px rgba(0, 150, 255, 0.4);
    }

    /* Blinking title animation */
    .blinking-title {
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0% { color: #ffcc00; text-shadow: 0 0 10px #ffcc00, 0 0 20px #ff4500, 0 0 30px #ff4500; }
        50% { color: #ff4500; text-shadow: 0 0 15px #ff4500, 0 0 25px #ffcc00, 0 0 35px #ff4500; }
        100% { color: #ffcc00; text-shadow: 0 0 10px #ffcc00, 0 0 20px #ff4500, 0 0 30px #ff4500; }
    }

    /* Stylish labels with 3D effect */
    label {
        font-weight: 600;
        color: #e0e0ff;
        text-shadow: 2px 2px 5px rgba(0, 100, 200, 0.5), -2px -2px 5px rgba(0, 150, 255, 0.3);
        font-size: 16px;
    }

    /* Input and selectbox styling with 3D effect */
    .stSelectbox, .stTextInput {
        background-color: rgba(0, 50, 100, 0.2);
        border: 1px solid rgba(0, 150, 255, 0.3);
        border-radius: 10px;
        color: #e0e0ff;
        box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.3), inset -2px -2px 5px rgba(0, 150, 255, 0.2);
    }
    .stSelectbox:hover, .stTextInput:hover {
        background-color: rgba(0, 50, 100, 0.3);
        box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.4), inset -2px -2px 5px rgba(0, 150, 255, 0.3);
    }

    /* Button styling with 3D glow */
    .stButton>button {
        background: linear-gradient(45deg, #00aaff, #00ffcc);
        color: #1a1a2e;
        border: none;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 16px;
        box-shadow: 0 3px 10px rgba(0, 170, 255, 0.5), 0 0 10px rgba(0, 255, 204, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0, 170, 255, 0.7), 0 0 15px rgba(0, 255, 204, 0.5);
        background: linear-gradient(45deg, #00ffcc, #00aaff);
    }

    /* Markdown and text styling with 3D effect */
    .stMarkdown, .stText {
        color: #e0e0ff !important;
        text-shadow: 1px 1px 3px rgba(0, 100, 200, 0.3), -1px -1px 3px rgba(0, 150, 255, 0.2);
    }

    /* Animation for output, weather icon, and highlight */
    .animated-output {
        animation: fadeIn 0.5s ease-in;
    }
    .weather-icon {
        animation: bounce 1.5s infinite;
    }
    .highlight-low {
        animation: glowGreen 2s infinite;
    }
    .highlight-medium {
        animation: glowOrange 2s infinite;
    }
    .highlight-high {
        animation: glowRed 2s infinite;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    @keyframes glowGreen {
        0% { box-shadow: 0 0 10px #2ECC71; }
        50% { box-shadow: 0 0 20px #2ECC71, 0 0 30px #2ECC71; }
        100% { box-shadow: 0 0 10px #2ECC71; }
    }
    @keyframes glowOrange {
        0% { box-shadow: 0 0 10px #F39C12; }
        50% { box-shadow: 0 0 20px #F39C12, 0 0 30px #F39C12; }
        100% { box-shadow: 0 0 10px #F39C12; }
    }
    @keyframes glowRed {
        0% { box-shadow: 0 0 10px #E74C3C; }
        50% { box-shadow: 0 0 20px #E74C3C, 0 0 30px #E74C3C; }
        100% { box-shadow: 0 0 10px #E74C3C; }
    }

    /* Severity trend gauge */
    .severity-gauge {
        height: 20px;
        border-radius: 10px;
        overflow: hidden;
        background-color: rgba(255, 255, 255, 0.1);
    }

    /* JavaScript-based clock */
    <script>
        function updateClock() {
            const now = new Date();
            const options = { hour: '2-digit', minute: '2-digit', hour12: true, timeZone: 'Asia/Kolkata' };
            const timeString = now.toLocaleTimeString('en-US', options) + ', ' + now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
            document.getElementById('clock').innerText = 'Time: ' + timeString;
        }
        setInterval(updateClock, 1000);
        updateClock();
    </script>
    <style>
        #clock {
            color: #e0e0ff;
            text-shadow: 2px 2px 5px rgba(0, 100, 200, 0.5), -2px -2px 5px rgba(0, 150, 255, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title with blinking animation ---
st.markdown(
    """
    <div style="text-align:center; animation: fadeIn 1s ease;">
        <h1 class="blinking-title" style="font-size: 2.5em;">
            🚦 Accident Severity Prediction
        </h1>
        <p style="font-size:20px; color:#e0e0ff; text-shadow: 2px 2px 5px rgba(0, 100, 200, 0.5), -2px -2px 5px rgba(0, 150, 255, 0.3);">
            Fill in the accident details below to get a severity prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Input Form with dynamic weather icon ---
with st.form("prediction_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        weather = st.selectbox("🌤️ Weather", ["Select Weather", "Clear", "Rainy", "Foggy", "Snowy"])
        weather_icon = "☀️" if weather == "Clear" else "🌧️" if weather == "Rainy" else "🌫️" if weather == "Foggy" else "❄️" if weather == "Snowy" else "🌤️"
        st.markdown(f'<div class="weather-icon">{weather_icon}</div>', unsafe_allow_html=True)
        time_of_day = st.selectbox("⏰ Time of Day", ["Select Time", "Morning", "Afternoon", "Evening", "Night"])
        traffic = st.selectbox("🚗 Traffic Level", ["Select Traffic", "Low", "Medium", "High"])

    with col2:
        road_condition = st.selectbox("🛣️ Road Condition", ["Select Road Condition", "Dry", "Wet", "Icy", "Snowy"])
        accident_type = st.selectbox("💥 Accident Type", ["Select Type", "Rear-end", "Head-on", "Side-impact", "Rollover"])

    submitted = st.form_submit_button("🔮 Predict Severity")

# --- Prediction Logic ---
if submitted:
    if (
        weather.startswith("Select") or
        road_condition.startswith("Select") or
        time_of_day.startswith("Select") or
        traffic.startswith("Select") or
        accident_type.startswith("Select")
    ):
        st.warning("⚠️ Please select all fields before predicting.", icon="⚠️")
    else:
        # Simulate "thinking" with a progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            progress_bar.progress(percent_complete + 1)

        data = CustomData(weather, road_condition, time_of_day, traffic, accident_type)
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)[0]

        # --- Stylish Animated Output ---
        severity_color = {"Low": "#2ECC71", "Medium": "#F39C12", "High": "#E74C3C"}
        st.markdown(
            f"""
            <div class="animated-output" style="background-color:{severity_color[results]};
                        padding:30px; border-radius:20px; text-align:center;
                        box-shadow: 0 5px 15px {severity_color[results]}, 0 0 20px rgba(0,0,0,0.5);
                        border: 2px solid rgba(255, 255, 255, 0.2);">
                <h2 style="color:#e0e0ff; text-shadow: 2px 2px 5px rgba(0, 100, 200, 0.5), -2px -2px 5px rgba(0, 150, 255, 0.3);
                           font-size: 2em;">🚨 Predicted Severity: {results}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Severity Trend Indicator (simulated) ---
        severity_trend = {"Low": 20, "Medium": 50, "High": 80}  # Percentage-based trend
        st.markdown(f'<div class="severity-gauge" style="width: 200px; margin: 0 auto;"><div style="width: {severity_trend[results]}%; height: 100%; background-color: {severity_color[results]};"></div></div>', unsafe_allow_html=True)
        st.caption(f"🌡️ Severity Trend: {severity_trend[results]}% (Simulated historical data)")

        # --- Highlight based on severity ---
        highlight_class = "highlight-low" if results == "Low" else "highlight-medium" if results == "Medium" else "highlight-high"
        highlight_icon = "✅" if results == "Low" else "🔔" if results == "Medium" else "🚨"
        st.markdown(
            f"""
            <div class="{highlight_class}" style="padding:20px; border-radius:15px; text-align:center; margin-top:10px;
                       background-color: rgba(255, 255, 255, 0.1);">
                <h3 style="color:#e0e0ff; text-shadow: 2px 2px 5px rgba(0, 100, 200, 0.5), -2px -2px 5px rgba(0, 150, 255, 0.3);
                          font-size: 1.5em;">{highlight_icon} {results} Alert!</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Risk Factors, Safety Recommendations, and Immediate Actions ---
        st.subheader("🛡️ Safety & Risk Analysis", divider="rainbow")
        confidence = random.randint(70, 95)  # Simulated confidence
        analysis_time = time.strftime("%I:%M %p IST, %B %d, %Y")  # Current time: 09:40 PM IST, August 24, 2025

        if results == "Low":
            st.success("✅ Low risk detected. Stay safe and drive responsibly!")
            st.markdown("""
                - 🌞 **Immediate Action:** Maintain current speed.
                - 🚦 **Immediate Action:** Keep safe distance from other vehicles.
                - 📋 **Recommendation:** Regularly check vehicle lights and tires.
                - 📊 **Risk Factor:** Confidence Level: {confidence}%
                - ⚖️ **Risk Factor:** Risk Level: {results}
                - ⏰ **Risk Factor:** Analysis Time: {analysis_time}
            """.format(confidence=confidence, results=results, analysis_time=analysis_time))
        elif results == "High":
            st.error("⚠️ High risk! Extreme caution advised.")
            st.markdown("""
                - 🚨 **Immediate Action:** Stop immediately if safe.
                - 📞 **Immediate Action:** Call emergency services at 911.
                - 🚫 **Recommendation:** Avoid driving until conditions improve.
                - 🛠️ **Recommendation:** Inspect vehicle for damage.
                - 📊 **Risk Factor:** Confidence Level: {confidence}%
                - ⚖️ **Risk Factor:** Risk Level: {results}
                - ⏰ **Risk Factor:** Analysis Time: {analysis_time}
            """.format(confidence=confidence, results=results, analysis_time=analysis_time))
        else:
            st.info("🟠 Moderate risk. Stay cautious on the road.")
            st.markdown("""
                - 🚗 **Immediate Action:** Reduce speed by 10-15 mph.
                - 👀 **Immediate Action:** Increase following distance.
                - 💡 **Recommendation:** Use headlights if visibility is poor.
                - 🚧 **Recommendation:** Avoid sudden lane changes.
                - 📊 **Risk Factor:** Confidence Level: {confidence}%
                - ⚖️ **Risk Factor:** Risk Level: {results}
                - ⏰ **Risk Factor:** Analysis Time: {analysis_time}
            """.format(confidence=confidence, results=results, analysis_time=analysis_time))

# --- Footer with JavaScript-based real-time clock ---
st.markdown(
    """
    <hr style="border-color: rgba(255, 255, 255, 0.2);">
    <p style="text-align:center; font-size:14px; color:#e0e0ff; text-shadow: 2px 2px 5px rgba(0, 100, 200, 0.5), -2px -2px 5px rgba(0, 150, 255, 0.3);
              animation: fadeIn 1s ease;">
    Made with ❤️ using <b>Streamlit</b> | Demo Accident Severity Predictor | <span id="clock"></span>
    </p>
    """,
    unsafe_allow_html=True,
)
