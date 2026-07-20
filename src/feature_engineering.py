import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ==========================================================
# File Paths
# ==========================================================
TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/val.csv"
TEST_PATH  = "data/processed/test.csv"

FEATURE_DIR = "data/feature"
MODEL_DIR   = "model"

TARGET_COLUMN = "annual_medical_cost"

# Drop IDs and columns not useful for prediction
DROP_COLUMNS = [
    "person_id","claims_count"
]

# ==========================================================
# Functions
# ==========================================================

def create_output_dirs():
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    return train_df, val_df, test_df

def clean_data(train_df, val_df, test_df):
    train_df = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
    val_df   = val_df.drop(columns=DROP_COLUMNS, errors="ignore")
    test_df  = test_df.drop(columns=DROP_COLUMNS, errors="ignore")
    return train_df, val_df, test_df

def separate_features_and_target(train_df, val_df, test_df):
    X_train, y_train = train_df.drop(columns=[TARGET_COLUMN]), train_df[TARGET_COLUMN]
    X_val, y_val     = val_df.drop(columns=[TARGET_COLUMN]), val_df[TARGET_COLUMN]
    X_test, y_test   = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_preprocessor(X_train):
    numerical_columns   = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(include=["object","string"]).columns.tolist()

    # Drop constant numeric columns (zero variance)
    constant_cols = [col for col in numerical_columns if X_train[col].nunique() <= 1]
    if constant_cols:
        print("Dropping constant columns:", constant_cols)
    numerical_columns = [col for col in numerical_columns if col not in constant_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),   # fill NaNs
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # fill NaNs
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns)
        ]
    )
    return preprocessor

def transform_and_save(preprocessor, X_train, y_train, X_val, y_val, X_test, y_test):
    # Fit on training data
    X_train = preprocessor.fit_transform(X_train)
    X_val   = preprocessor.transform(X_val)
    X_test  = preprocessor.transform(X_test)

    # Save preprocessing object
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessing.pkl"))

    # Convert to DataFrames
    def to_dataframe(X):
        if hasattr(X, "toarray"):
            return pd.DataFrame(X.toarray())
        return pd.DataFrame(X)

    X_train = to_dataframe(X_train)
    X_val   = to_dataframe(X_val)
    X_test  = to_dataframe(X_test)

    # Save features and targets
    X_train.to_csv(os.path.join(FEATURE_DIR, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(FEATURE_DIR, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(FEATURE_DIR, "X_test.csv"), index=False)

    y_train.to_csv(os.path.join(FEATURE_DIR, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(FEATURE_DIR, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(FEATURE_DIR, "y_test.csv"), index=False)

    print("Feature Engineering Completed Successfully")

# ==========================================================
# Main Execution
# ==========================================================
# ==========================================================
# Main Execution
# ==========================================================

if __name__ == "__main__":

    print("Starting Feature Engineering...")

    create_output_dirs()

    train_df, val_df, test_df = load_data()

    print("Datasets Loaded Successfully")

    train_df, val_df, test_df = clean_data(
        train_df,
        val_df,
        test_df
    )

    print("Unnecessary Columns Removed")

    X_train, y_train, X_val, y_val, X_test, y_test = separate_features_and_target(
        train_df,
        val_df,
        test_df
    )

    print("Features and Target Separated")

    preprocessor = build_preprocessor(X_train)

    print("Preprocessor Created")

    transform_and_save(
        preprocessor,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    print("All files saved successfully.")