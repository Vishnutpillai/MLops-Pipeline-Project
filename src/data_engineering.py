import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


raw_data_path = "data/raw/medical_insurance.csv"
processed_data="data/processed"

def process_data(df):
    """
    perform basic data cleaning and preprocessing on the input DataFrame.
    """
    
    print(f"original shape of the data: {df.shape}")

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print("\nMissing values before cleaning:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object", "string"]).columns

        # Fill missing values
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median(), inplace=True) 
        for col in categorical_cols:
            df[col] = df[col].fillna("unknown", inplace=True)   

        print(f"\nMissing values after cleaning:: {df.isnull().sum().sum()}")
    
    else:
        print("\nNo missing values found in the dataset.") 

    return df

#split the data into train and test sets
def split_data(df):
    """
    Split the input DataFrame into training, testing sets and validation set.
    """
    train, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val = train_test_split(train, test_size=0.15, random_state=42)
    print(f"\nTraining set shape: {train.shape}")
    print(f"Testing set shape: {test.shape}")
    print(f"Validation set shape: {val.shape}")

    return train, test, val

def save_dataset(train,test,val):
    """
    Save the training, testing and validation sets to CSV files.
    """
    os.makedirs(processed_data, exist_ok=True)
    train.to_csv(os.path.join(processed_data, "train.csv"), index=False)
    test.to_csv(os.path.join(processed_data, "test.csv"), index=False)
    val.to_csv(os.path.join(processed_data, "val.csv"), index=False)
    print(f"\nDatasets saved to {processed_data} directory.")


if __name__ == "__main__":
    # Load the raw data
    df = pd.read_csv(raw_data_path)

    # Process the data
    df = process_data(df)

    # Split the data into training, testing and validation sets
    train, test, val = split_data(df)

    # Save the datasets to CSV files
    save_dataset(train, test, val)

