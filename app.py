from flask import Flask, render_template, request
import pandas as pd
import random # We'll use this to simulate a prediction

# --- Start of placeholder classes to make the app runnable ---
# In a real application, these would be in 'src/pipeline/predict_pipeline.py'
class CustomData:
    """
    This class is a placeholder for your actual CustomData class.
    It's designed to fix the 'weather' vs 'Weather' error by ensuring
    all parameter names are consistent with the HTML form inputs.
    """
    def __init__(self, weather, road_condition, time_of_day, traffic, accident_type):
        self.weather = weather
        self.road_condition = road_condition
        self.time_of_day = time_of_day
        self.traffic = traffic
        self.accident_type = accident_type

    def get_data_as_data_frame(self):
        """
        Converts the instance data into a pandas DataFrame, a common
        format for machine learning models.
        """
        custom_data_input_dict = {
            'weather': [self.weather],
            'road_condition': [self.road_condition],
            'time_of_day': [self.time_of_day],
            'traffic': [self.traffic],
            'accident_type': [self.accident_type]
        }
        return pd.DataFrame(custom_data_input_dict)

class PredictPipeline:
    """
    This is a placeholder for your actual prediction pipeline.
    It simulates a prediction based on the input data.
    """
    def predict(self, features):
        """
        Simulates the prediction of accident severity.
        In a real scenario, this would load a model and make a prediction.
        """
        # For demonstration, we'll return a random severity from a list.
        severities = ["Low", "Medium", "High"]
        # The 'features' DataFrame isn't used here, but it would be in a real model.
        return [random.choice(severities)]
# --- End of placeholder classes ---

# Initialize Flask app
application = Flask(__name__)
app = application

# Home route
@app.route('/')
def index():
    return render_template('index.html')  # Main landing page

# Prediction route
@app.route('/predict_accident', methods=['GET', 'POST'])
def predict_accident():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # Collect form data using the correct, consistent names.
        data = CustomData(
            weather=request.form.get('weather'),
            road_condition=request.form.get('road_condition'),
            time_of_day=request.form.get('time_of_day'),
            traffic=request.form.get('traffic'),
            accident_type=request.form.get('accident_type')
        )

        # Convert form data to DataFrame
        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:")
        print(pred_df)

        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print("Prediction Results:", results)

        # Pass results to HTML for dynamic display
        return render_template('index.html', results=results[0])

if __name__ == "__main__":
    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
