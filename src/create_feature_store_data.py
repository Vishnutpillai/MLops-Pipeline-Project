import os
import pandas as pd

# Input files
PROCESSED_PATH = "data/processed/train.csv"
FEATURE_PATH = "data/feature/X_train.csv"

# Output directory
OUTPUT_DIR = "data/features_store"
CSV_FILE = "train_features.csv"
PARQUET_FILE = "train_features.parquet"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read input files
processed_df = pd.read_csv(PROCESSED_PATH)
feature_df = pd.read_csv(FEATURE_PATH)

print("Processed shape :", processed_df.shape)
print("Feature shape   :", feature_df.shape)

# Create feature store dataframe
feature_store_df = pd.concat(
    [processed_df[["person_id"]], feature_df],
    axis=1
)

# Add event timestamp
feature_store_df["event_timestamp"] = pd.Timestamp.now()

# Save CSV
csv_path = os.path.join(OUTPUT_DIR, CSV_FILE)
feature_store_df.to_csv(csv_path, index=False)

# Save Parquet
parquet_path = os.path.join(OUTPUT_DIR, PARQUET_FILE)
feature_store_df.to_parquet(parquet_path, index=False)

print("\nFeature Store Dataset Created Successfully!")
print(f"CSV      : {csv_path}")
print(f"Parquet  : {parquet_path}")

print("\nColumns:")
print(feature_store_df.columns.tolist())

print("\nPreview:")
print(feature_store_df.head())