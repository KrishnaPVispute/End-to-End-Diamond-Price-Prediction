import pandas as pd
import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor  # FIX: Changed from Classifier → Regressor
from src.exception import customexception
from src.logger import logging 
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    # Path where best trained model will be saved
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent Variable from train and test data")

            # FIX: Corrected slicing for y_train (previously used :-1 by mistake)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],   # all columns except last → features
                train_array[:, -1],    # last column → target
                test_array[:, :-1],
                test_array[:, -1]   
            )

            # Define models to compare
            models = {
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elasticnet': ElasticNet(),
                'DecisionTree': DecisionTreeRegressor()  # FIX: Regressor instead of Classifier
            }

            # Train and evaluate models
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("------------------------------------------------------------------------------------------------")
            logging.info(f"Model Report:{model_report}")

            # Get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # FIX: Corrected .key() → .keys()
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # Log best model
            print(f'Best Model Found, Model Name :{best_model_name}, R2 Score:{best_model_score}')
            print("---------------------------------------------------------------------------------------------")
            logging.info(f"Best Model Found, Model Name {best_model_name}, with R2 score {best_model_score} ")

            # Save best model to artifacts
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise customexception(e, sys)
