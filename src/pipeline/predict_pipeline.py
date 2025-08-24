import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        Weather: str,
        Road_Condition: str,
        Time_of_Day: str,
        Traffic: str,
        Accident_Type: str,
        Vehicle_Type: str,
        Accident_Reason: str,
        Latitude: float,
        Longitude: float):

        self.Weather = Weather
        self.Road_Condition = Road_Condition
        self.Time_of_Day = Time_of_Day
        self.Traffic = Traffic
        self.Accident_Type = Accident_Type
        self.Vehicle_Type = Vehicle_Type
        self.Accident_Reason = Accident_Reason
        self.Latitude = Latitude
        self.Longitude = Longitude

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Weather": [self.Weather],
                "Road_Condition": [self.Road_Condition],
                "Time_of_Day": [self.Time_of_Day],
                "Traffic": [self.Traffic],
                "Accident_Type": [self.Accident_Type],
                "Vehicle_Type": [self.Vehicle_Type],
                "Accident_Reason": [self.Accident_Reason],
                "Latitude": [self.Latitude],
                "Longitude": [self.Longitude],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
