

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import joblib

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)




@app.get("/")
async def root():
    return {"Status": "Up and running"}


# Implement a /predict endpoint

@app.get("/predict")
def predict(measure_index, measure_moisture, measure_temperature, measure_chemicals, measure_biodiversity, main_element, soil_condition, datetime_start, datetime_end
):
    # turn values into the right types
    measure_index = float(measure_index)
    measure_moisture = float(measure_moisture)
    measure_temperature = float(measure_temperature)
    measure_chemicals = float(measure_chemicals)
    measure_biodiversity = float(measure_biodiversity)
    datetime_start = datetime.strptime(datetime_start, "%Y-%m-%d %H:%M:%S")
    datetime_end = datetime.strptime(datetime_end, "%Y-%m-%d %H:%M:%S")
    
    request_params = {
        "measure_index": [measure_index],
        "measure_moisture": [measure_moisture],
        "measure_temperature": [measure_temperature],
        "measure_chemicals": [measure_chemicals],
        "measure_biodiversity": [measure_biodiversity],
        "main_element": main_element,
        "soil_condition": soil_condition,
        "datetime_start": [datetime_start],
        "datetime_end": [datetime_end],
    }
    
    X_pred = pd.DataFrame(request_params)
    pipeline = joblib.load("assets/model.joblib")
    pred = pipeline.predict(X_pred)
    
    return{"viable": bool(pred)}