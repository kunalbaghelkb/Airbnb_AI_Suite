import os
import sys
import pickle
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sqlalchemy import create_engine
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def get_database_connection():
    try:
        # Getting path of Data Folder
        db_path = os.path.join('data','database')
        
        # If not exist then create one
        os.makedirs(db_path, exist_ok=True)
        
        # Database File name
        db_file = os.path.join(db_path, 'airbnb.db')
        
        # Connection String for SQLite
        connection_string = f'sqlite:///{db_file}'
        
        # Create Engine
        engine = create_engine(connection_string)
        
        logging.info(f"Database connection established at {db_file}")
        return engine
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    '''
    Serilization/Pickling: To Save Python Object (Model, Preprocessor) in file
    
    Args:
        file_path: Where to save?
        obj: What to save? (e.g., trained model)
    '''
    
    try:
        dir_path = os.path.dirname(file_path)
        
        # Create Folder if not exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Open file in write mode and save it
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Object saved successfully at {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    '''
    Load Objects
    '''
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    '''
    This will train all models one by one
    and Hyperparameter tuning will also be done (GridSearchCV)
    '''
    try:
        report = {}
        
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]
            
            # use GridSearchCV for best params
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            # Set models with best params
            model.set_params(**gs.best_params_)
            
            # Train Final Model
            model.fit(X_train, y_train)
            
            # Prediction
            y_test_pred = model.predict(X_test)
            
            # Metrics
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            
            logging.info(f"Model: {model_name} | R2 Score: {test_model_score}")
        return report
            
            
    except Exception as e:
        raise CustomException(e, sys)