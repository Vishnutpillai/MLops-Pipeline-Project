import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset


REPORT_DIR = "monitoring/reports"
REPORT_FILE = os.path.join(REPORT_DIR, "data_drift.html")


def generate_drift_report(reference_path, current_path):
    """
    Compare reference and current datasets
    and generate an HTML data drift report.
    """

    os.makedirs(REPORT_DIR, exist_ok=True)

    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        current_data=current_data,
        reference_data=reference_data
    )

    result.save_html(REPORT_FILE)

    print(f"Drift report created: {REPORT_FILE}")


if __name__ == "__main__":

    reference_path = "data/processed/train.csv"
    current_path = "data/processed/val.csv"

    generate_drift_report(
        reference_path,
        current_path
    )