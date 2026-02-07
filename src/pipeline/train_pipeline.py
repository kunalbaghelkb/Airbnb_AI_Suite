import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.genai_engine import GenAIEngine
from src.exception import CustomException

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        """
        Ye function poora training process automate karta hai:
        1. Data Ingestion
        2. Transformation
        3. Model Training
        4. GenAI Vector DB Creation
        """
        try:
            print("🚀 Training Pipeline Started...")

            # --- Step 1: Data Ingestion ---
            print("\n🔵 Step 1: Data Ingestion Started")
            ingestion_obj = DataIngestion()
            train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()
            print("✅ Data Ingestion Completed")

            # --- Step 2: Data Transformation ---
            print("\n🔵 Step 2: Data Transformation Started")
            transform_obj = DataTransformation()
            train_arr, test_arr, _ = transform_obj.initiate_data_transformation(train_data_path, test_data_path)
            print("✅ Data Transformation Completed")

            # --- Step 3: Model Training ---
            print("\n🔵 Step 3: Model Training Started")
            trainer_obj = ModelTrainer()
            r2_square = trainer_obj.initiate_model_trainer(train_arr, test_arr)
            print(f"✅ Model Training Completed. Best R2 Score: {r2_square}")

            # --- Step 4: GenAI Vector DB Creation ---
            print("\n🔵 Step 4: GenAI Engine (Vector DB) Creation Started")
            genai_obj = GenAIEngine()
            status = genai_obj.create_vector_db()
            print(f"{status}")

            print("\n🏆 All Training Pipelines Completed Successfully!")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()