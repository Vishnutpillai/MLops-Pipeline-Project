import os
import joblib
import pandas as pd

# ==========================
# Load Preprocessor
# ==========================

PREPROCESSOR_PATH = os.path.join(
    "model",
    "preprocessing.pkl"
)

preprocessor = joblib.load(PREPROCESSOR_PATH)

# ==========================
# Load Best Model
# ==========================

MODEL_PATH = os.path.join(
    "model",
    "best_model.pkl"      # Change if your filename is different
)

model = joblib.load(MODEL_PATH)

# ==========================
# Prediction Function
# ==========================

def predict_cost(data):

    # Convert API input to dictionary
    input_data = data.model_dump()

    # Convert dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Apply preprocessing
    transformed_data = preprocessor.transform(df)

    # Predict
    prediction = model.predict(transformed_data)

    # Return prediction
    return float(prediction[0])