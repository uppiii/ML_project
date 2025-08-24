import sys
import os
# Ensure project root is in sys.path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from dataclasses import dataclass
import os
import joblib

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Placeholder for custom modules. Replace with your actual imports.
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class to store the path for the preprocessor object.
    The preprocessor will be saved in the 'artifacts' directory.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "severity_preprocessor.pkl")

class DataTransformation:
    """
    This class is responsible for the entire data transformation process.
    It includes methods for creating a data preprocessor and initiating
    the transformation on training and testing data.
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns the data preprocessor object.
        This function handles feature engineering and scaling for both
        numerical and categorical features identified from your dataset.
        """
        try:
            # Columns identified from the provided dataset
            numerical_columns = ["Latitude", "Longitude"]
            categorical_columns = [
                "Weather",
                "Road_Condition",
                "Time_of_Day",
                "Traffic",
                "Accident_Type",
                "Vehicle_Type",
                "Accident_Reason",
            ]
            
            # Numerical pipeline to handle missing values and scale data
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline to handle missing values, one-hot encode, and scale
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # ColumnTransformer to apply the correct pipeline to each column type
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipelines", categorical_pipeline, categorical_columns)
                ],
                remainder='drop' # Drops any columns not specified in the pipelines
            )

            return preprocessor
        
        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process by loading,
        preprocessing, and combining the data.
        
        Args:
            train_path (str): The file path to the training dataset.
            test_path (str): The file path to the testing dataset.

        Returns:
            tuple: A tuple containing the transformed training array,
                   transformed testing array, and the path to the
                   saved preprocessor object.
        """
        try:
            # Read the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            logging.info("Obtaining preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Severity"

            # Separate features and target from the dataframes
            # Note: 'Date' and 'Time' columns are dropped as they are not used in this transformation pipeline.
            input_feature_train_df = train_df.drop(columns=[target_column_name, "Date", "Time"], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name, "Date", "Time"], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing data.")

            # Apply the transformations using fit_transform on train data and transform on test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine transformed features with the target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Transformed arrays created successfully.")
            
            # Save the preprocessor object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {e}")
            raise CustomException(e, sys)