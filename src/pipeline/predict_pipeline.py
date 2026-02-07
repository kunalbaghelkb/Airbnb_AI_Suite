import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        '''
        This Function will get data from frontend,
        it will do preprocess and it will predict
        '''
        try:
            # Load Artifacts
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            print("Loading Model and Preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Step 1: Transform Data (Scaling + OneHotEncoding)
            data_scaled = preprocessor.transform(features)
            
            # Step 2: Do prediction
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
        neighbourhood_group: str,
        neighbourhood: str,
        latitude: float,
        longitude: float,
        room_type: str,
        minimum_nights: int,
        number_of_reviews: int,
        reviews_per_month: float,
        calculated_host_listings_count: int,
        availability_365: int):
        
        self.neighbourhood_group = neighbourhood_group
        self.neighbourhood = neighbourhood
        self.latitude = latitude
        self.longitude = longitude
        self.room_type = room_type
        self.minimum_nights = minimum_nights
        self.number_of_reviews = number_of_reviews
        self.reviews_per_month = reviews_per_month
        self.calculated_host_listings_count = calculated_host_listings_count
        self.availability_365 = availability_365
        
    def get_data_as_data_frame(self):
        '''
        This Function will convert data to DataFrame
        so model can read.
        '''
        try:
            custom_data_input_dict = {
                "neighbourhood_group": [self.neighbourhood_group],
                "neighbourhood": [self.neighbourhood],
                "latitude": [self.latitude],
                "longitude": [self.longitude],
                "room_type": [self.room_type],
                "minimum_nights": [self.minimum_nights],
                "number_of_reviews": [self.number_of_reviews],
                "reviews_per_month": [self.reviews_per_month],
                "calculated_host_listings_count": [self.calculated_host_listings_count],
                "availability_365": [self.availability_365],
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)