from fastapi import FastAPI
from prometheus_client import make_asgi_app
import time

from app.schemas import InsuranceInput
from app.predict import predict_cost

from monitoring.logger import log_prediction
from monitoring.health import get_system_health
from monitoring.metrics import (
    record_prediction,
    record_api_request,
    observe_prediction_latency
)


app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    version="1.0.0"
)


# -----------------------------
# Prometheus Metrics
# -----------------------------
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# -----------------------------
# Request Monitoring Middleware
# -----------------------------
@app.middleware("http")
async def monitor_requests(request, call_next):

    record_api_request()

    response = await call_next(request)

    return response


# -----------------------------
# Home Endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Medical Insurance Cost Prediction API is Running"
    }


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health():
    return get_system_health()


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: InsuranceInput):

    start_time = time.time()

    prediction = predict_cost(data)

    elapsed_time = time.time() - start_time

    # Record prediction metric
    record_prediction()

    # Record prediction latency
    observe_prediction_latency(elapsed_time)

    # Convert input to dictionary
    try:
        input_data = data.model_dump()
    except AttributeError:
        input_data = data.dict()

    # Save prediction log
    log_prediction(
        input_data=input_data,
        prediction=prediction
    )

    return {
        "Predicted Annual Medical Cost": prediction
    }