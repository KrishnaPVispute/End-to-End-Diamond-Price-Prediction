import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import customexception
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        # Save model object in pickle file
        with open(file_path, 'wb') as file_obj:  # Write byte mode(WB)
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    """
    Trains multiple models and evaluates them using R2 score.
    Returns a dictionary with model name as key and score as value.
    """
    try:
        # Ensure y_train and y_test are 1D arrays (important for sklearn metrics)
        if isinstance(y_train, (pd.DataFrame, pd.Series)):
            y_train = y_train.values.ravel()
        else:
            y_train = np.ravel(y_train)

        if isinstance(y_test, (pd.DataFrame, pd.Series)):
            y_test = y_test.values.ravel()
        else:
            y_test = np.ravel(y_test)

        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Train model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # If predictions are shaped (n,1), flatten to (n,)
            if y_test_pred.ndim > 1 and y_test_pred.shape[1] == 1:
                y_test_pred = y_test_pred.ravel()

            # Calculate R2 score
            test_model_score = r2_score(y_test, y_test_pred)

            # Save result in report dict
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        logging.info("Exception Occurred during model Training")
        raise customexception(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)