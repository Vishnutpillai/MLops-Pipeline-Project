import os
import pandas as pd

PROCESSED_PATH = "data/processed/train.csv"
FEATURE_PATH = "data/feature/X_train.csv"
OUTPUT_DIR = "data/features_store"
OUTPUT_FILE = "train_features.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

processed_df=pd.read_csv(PROCESSED_PATH)
feature_df=pd.read_csv(FEATURE_PATH)
print("Processed shape:", processed_df.shape)
print("Feature shape:", feature_df.shape)

person = processed_df[["person_id"]]

feature_store_df = pd.concat([person, feature_df], axis=1)

#add current timestamp
feature_store_df["event_timestamp"] = pd.Timestamp.now()

#save

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
feature_store_df.to_csv(output_path, index=False)

print("Feature Store Dataset Created Successfully")

print(feature_store_df.head(10))