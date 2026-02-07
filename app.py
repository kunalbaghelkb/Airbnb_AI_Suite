import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['OMP_NUM_THREADS'] = '1'
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.components.genai_engine import GenAIEngine

# 1. App Initialize
app = FastAPI(
    title="Airbnb AI Suite API",
    description="API for Price Prediction & GenAI Search",
    version="1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Engine
genai_engine = GenAIEngine()
predict_pipeline = PredictPipeline()

# Health Check Route
@app.get("/")
def home():
    return {"message": "Airbnb AI Engine is Running!"}

# GenAI Search Route
@app.get("/search")
def search_listings(query: str):
    '''
    Example: /search?query=apartment near park
    '''
    results = genai_engine.search_listings(query)
    return {"results": results}

# Price Prediction Route
class ListingInput(BaseModel):
    neighbourhood_group: str
    neighbourhood: str
    latitude: float
    longitude: float
    room_type: str
    minimum_nights: int
    number_of_reviews: int
    reviews_per_month: float
    calculated_host_listings_count: int
    availability_365: int
    
@app.post("/predict")
def predict_price(input_data: ListingInput):
    # 1. Convert Input JSON to Object
    data = CustomData(
        neighbourhood_group=input_data.neighbourhood_group,
        neighbourhood=input_data.neighbourhood,
        latitude=input_data.latitude,
        longitude=input_data.longitude,
        room_type=input_data.room_type,
        minimum_nights=input_data.minimum_nights,
        number_of_reviews=input_data.number_of_reviews,
        reviews_per_month=input_data.reviews_per_month,
        calculated_host_listings_count=input_data.calculated_host_listings_count,
        availability_365=input_data.availability_365
    )
    
    # 2. Convert to DataFrame
    df = data.get_data_as_data_frame()
    
    # 3. Predict
    pred_log = predict_pipeline.predict(df)
    
    # 4. Inverse Log (Original Price)
    import numpy as np
    final_price = np.expm1(pred_log[0])
    
    return {"predicted_price": float(final_price)}

# Run Server (For Debugging)
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)