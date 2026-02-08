
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load

# Initialize app
app = FastAPI()

# Load model and preprocessor
model = load("model.joblib")
preprocessor = load("preprocessor.joblib")

# Input data schema
class InputData(BaseModel):
    Age: float
    TypeofContact: str
    CityTier: int
    DurationOfPitch: float
    Occupation: str
    Gender: str
    NumberOfPersonVisiting: int
    NumberOfFollowups: float
    ProductPitched: str
    PreferredPropertyStar: float
    MaritalStatus: str
    NumberOfTrips: float
    Passport: int
    PitchSatisfactionScore: int
    OwnCar: int
    NumberOfChildrenVisiting: float
    Designation: str
    MonthlyIncome: float

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
