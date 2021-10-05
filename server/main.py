# 1. Library imports
import uvicorn
from fastapi import FastAPI
import json 

from app import resume_predict

# 2. Create app and model objects
app = FastAPI()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict_species(content):
    data = content
    prediction = resume_predict(data)

    return prediction
