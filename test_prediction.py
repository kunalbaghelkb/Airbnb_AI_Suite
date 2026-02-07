from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Fake Data to Verify
data = CustomData(
    neighbourhood_group='Manhattan',
    neighbourhood='Harlem',
    latitude=40.82085,
    longitude=73.94025,
    room_type='Private room',
    minimum_nights=3,
    number_of_reviews=50,
    reviews_per_month=1.5,
    calculated_host_listings_count=1,
    availability_365=200
)

# Convert into DataFrame
df = data.get_data_as_data_frame()
print("Input DataFrame:")
print(df)

# Call Prediction Pipeline
pipeline = PredictPipeline()
prediction = pipeline.predict(df)

# Result (Recall: we did log transformation, so we need to get exponent again for original price)
import numpy as np
final_price = np.expm1(prediction[0])

print(f"\n Predicted Price: ${final_price:.2f}")