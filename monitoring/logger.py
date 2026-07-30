import csv
import os
from datetime import datetime


LOG_DIR = "monitoring/logs"
LOG_FILE = os.path.join(LOG_DIR, "predictions.csv")


def log_prediction(input_data, prediction, probability=None):
    """
    Save a prediction to monitoring/logs/predictions.csv
    """

    os.makedirs(LOG_DIR, exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "prediction",
                "probability",
                "input_data"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            prediction,
            probability,
            str(input_data)
        ])