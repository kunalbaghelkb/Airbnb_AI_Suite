import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        Pipeline Logic:
        1. Numerical (Standard) -> Mean Impute + Scaling
        2. Numerical (Reviews) -> Constant 0 Impute + Scaling
        3. Categorical -> Mode Impute + OneHotEncoding
        '''
        try:
            # Define Column Groups based on EDA
            numerical_columns = [
                "latitude",
                "longitude",
                "minimum_nights",
                "number_of_reviews",
                "calculated_host_listings_count",
                "availability_365"
            ]
            
            # Special Case: if review is null then consider it to 0
            reviews_columns = ["reviews_per_month"]
            
            categorical_columns = [
                "neighbourhood_group",
                "room_type",
                "neighbourhood" # 220 Categories, OHE will handle it
            ]
            
            # Pipeline 1: Standard Numerical Columns
            num_pipeline = Pipeline(
                steps=[
                    # There may be outliers, so median will be safe
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Pipeline 2: Review Specific (Fill NaN with 0)
            reviews_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                    ("scaler", StandardScaler())
                ]
            )
            
            # Pipeline 3: Categorical Columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False)) # After OHE Scaling is Optional
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Combine all pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("reviews_pipeline", reviews_pipeline, reviews_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "price"
            
            # Drop Unwanted Columns
            drop_columns = [target_column_name, "id", "name", "host_id", "host_name", "last_review"]
            
            # Split Features and Target
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            # Main Transformation
            # fit_transform on train data while transform on test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Log Transformation on Target Variable
            target_feature_train_df = np.log1p(target_feature_train_df)
            target_feature_test_df = np.log1p(target_feature_test_df)
            
            # Combining Train and Test array to save
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")
            
            # Used Utilize function to save pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)