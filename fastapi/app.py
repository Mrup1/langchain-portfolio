from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd
import numpy as np # Import numpy to handle potential numpy types

# import the ml model
# Ensure 'model.pkl' is in the same directory as this script
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl not found. Please ensure the model file is in the same directory.")
    # Exit or handle the error appropriately if the model is crucial
    model = None # Set model to None or a dummy for testing if needed

app = FastAPI()

class InputData(BaseModel):
    id : int = Field(description="Unique identifier for the input data")
    gender: Literal['Male', 'Female']
    age: float
    hypertension: int
    heart_disease: int
    ever_married: Literal['Yes', 'No']
    work_type: Literal['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    Residence_type: Literal['Urban', 'Rural']
    avg_glucose_level: float
    bmi: float
    smoking_status: Literal['formerly smoked', 'never smoked', 'smokes']


@app.post("/predict")
def predict(input_data: InputData):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "ML model not loaded."})

    # Preprocess the input data
    # Create a DataFrame from the input_data
    df = pd.DataFrame([
        {
            "id": input_data.id,
            "gender": input_data.gender,
            "age": input_data.age,
            "hypertension": input_data.hypertension,
            "heart_disease": input_data.heart_disease,
            "ever_married": input_data.ever_married,
            "work_type": input_data.work_type,
            "Residence_type": input_data.Residence_type,
            "avg_glucose_level": input_data.avg_glucose_level,
            "bmi": input_data.bmi,
            "smoking_status": input_data.smoking_status,
        }
    ])

    # Make the prediction
    prediction = model.predict(df)
    probs = model.predict_proba(df)[0] 
    serializable_prediction = int(prediction[0])
    confidence = float(np.max(probs))
    class_probs = {f"class_{i}": float(prob) for i, prob in enumerate(probs)}

    return JSONResponse(
        status_code=200,
        content={
            "response": {
                "predicted_category": serializable_prediction,
                "confidence": round(confidence, 3),
                "class_probabilities": class_probs
            }
        }
    )