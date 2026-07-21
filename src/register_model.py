import json
import joblib
import mlflow
import mlflow.xgboost

# ==============================
# MLflow Tracking Server
# ==============================

# Point to your local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Create / Select experiment
mlflow.set_experiment("Insurance Pricing Model")


# ==============================
# Load trained model
# ==============================

model = joblib.load("model/best_model.pkl")


# ==============================
# Load Metrics & Hyperparameters
# ==============================

with open("model/evaluation/metrics.json") as f:
    metrics = json.load(f)

with open("model/evaluation/best_params.json") as f:
    params = json.load(f)

best_model_name = params["best_model"]
best_params = params["best_params"]
model_metrics = metrics[best_model_name]


# ==============================
# MLflow Run
# ==============================

with mlflow.start_run(run_name=best_model_name):

    # Log model name
    mlflow.log_param("model_name", best_model_name)

    # Log hyperparameters
    for key, value in best_params.items():
        mlflow.log_param(key, value)

    # Log metrics
    for key, value in model_metrics.items():
        mlflow.log_metric(key, value)

    # Register Model (Using XGBoost flavor)
    mlflow.xgboost.log_model(
    xgb_model=model,
    name="model",
    registered_model_name="InsurancePricingModel"
    )
print("Model Registered Successfully")