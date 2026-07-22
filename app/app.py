from fastapi import FastAPI

from app.schemas import InsuranceInput
from app.predict import predict_cost

app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Medical Insurance Cost Prediction API is Running"
    }


@app.post("/predict")
def predict(data: InsuranceInput):

    prediction = predict_cost(data)

    return {
        "Predicted Annual Medical Cost": prediction
    }