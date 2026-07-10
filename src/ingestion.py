import os
import pandas as pd
import boto3

BUCKET_NAME = "mlops-insurance-project-data"
OBJECT_KEY = "raw/medical_insurance.csv"

local_dir ="data/raw"
local_file = os.path.join(local_dir, "medical_insurance.csv")

def load_from_s3():
    
    os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client("s3")
    print("downloading file from S3...")
    
    s3.download_file(BUCKET_NAME, OBJECT_KEY, local_file)
    print("File downloaded successfully.")
    df = pd.read_csv(local_file)
    print(f"dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    df = load_from_s3()
    