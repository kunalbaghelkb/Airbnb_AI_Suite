import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import get_database_connection

# 1. Configuration
@dataclass
class DataIngestionConfig:
    # Will Create this 3 files in Artifacts folder
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")
    
# 2. Data Ingestion Process
class DataIngestion:
    def __init__(self):
        # Initialize Config to get file path
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # 1. Reading Raw Data
            df = pd.read_csv('data/raw/listings.csv')
            logging.info('Read the dataset as dataframe from Raw CSV')
            
            # 2. Database Initialize
            engine = get_database_connection()
            df.to_sql('listings_raw', engine, if_exists='replace', index=False)
            logging.info('Data pushed to SQLite Database successfully')
            
            # 3. Reading from Database
            df_new = pd.read_sql('SELECT * FROM listings_raw', engine)
            logging.info('Fetched data back from Database for processing')
            
            # 4. Saving Raw File to Artifacts (For Backup)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            
            # 5. Train Test Split
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df_new, test_size=0.2, random_state=42)
            
            # 6. Saving Split Files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f"Ingestion Data Completed. Train shape: {train_set.shape}, Test shape: {test_set.shape}")
            
            # Return paths
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
