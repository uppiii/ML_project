import os
import sys
from dataclasses import dataclass
import pickle # Added this import to explicitly show it's used
import pandas as pd # You'll need this for data
from sklearn.model_selection import train_test_split # Used for splitting data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    # This path is where the .pkl file will be saved
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Splitting training and test input data")

            # Define candidate models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=200),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "SVM": SVC(),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
                "CatBoost": CatBoostClassifier(verbose=0),
            }

            # Define hyperparameter grids
            params = {
                "Logistic Regression": {
                    "C": [0.1, 1, 10]
                },
                "Decision Tree": {
                    "max_depth": [3, 5, 7, None]
                },
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10, None]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "SVM": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5, 7]
                },
                "CatBoost": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [3, 5, 7]
                }
            }

            # Evaluate models
            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best Model Found: {best_model_name} with Accuracy: {best_score}")

            # Save best model using the save_object function
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)

# --- This is the code you need to add to the bottom of the file ---
if __name__ == "__main__":
    import numpy as np
    
    # Generate more realistic dummy data with sufficient samples
    np.random.seed(42)
    n_samples = 100  # Increased from 6 to 100 samples
    
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.choice(['X', 'Y'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    df = pd.DataFrame(data)
    
    # Simple one-hot encoding for the categorical features
    df = pd.get_dummies(df, columns=['feature2', 'feature4'], drop_first=True)
    
    # Prepare data for training
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Now, initiate the model trainer
    model_trainer = ModelTrainer()
    best_model_name, best_score = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
    
    print(f"Model training process completed. Best model saved: {best_model_name} with score: {best_score}")
