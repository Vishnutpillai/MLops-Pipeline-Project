import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================================
# File Paths
# ==========================================================

feature_dir = "data/feature"
model_dir = "model"
evaluation_dir = "model/evaluation"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(evaluation_dir, exist_ok=True)

x_train = pd.read_csv(os.path.join(feature_dir, "X_train.csv"))
y_train = pd.read_csv(os.path.join(feature_dir, "y_train.csv")).squeeze()

x_val = pd.read_csv(os.path.join(feature_dir, "X_val.csv"))
y_val = pd.read_csv(os.path.join(feature_dir, "y_val.csv")).squeeze()

x_test = pd.read_csv(os.path.join(feature_dir, "X_test.csv"))
y_test = pd.read_csv(os.path.join(feature_dir, "y_test.csv")).squeeze()

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print("Train Shape :", x_train.shape)
print("Validation Shape :", x_val.shape)
print("Test Shape :", x_test.shape)

# ==========================================================
# Random Forest
# ==========================================================

print("\nTraining Random Forest...\n")

rf = RandomForestRegressor(random_state=42)

rf_params = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_params,
    n_iter=10,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

rf_search.fit(x_train, y_train)

best_rf = rf_search.best_estimator_

rf_pred = best_rf.predict(x_val)

rf_mse = mean_squared_error(y_val, rf_pred)
rf_mae = mean_absolute_error(y_val, rf_pred)
rf_r2 = r2_score(y_val, rf_pred)

print("Random Forest Completed")

# ==========================================================
# XGBoost
# ==========================================================

print("\nTraining XGBoost...\n")

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist"
)

xgb_params = {
    "n_estimators":[300,500,700],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "gamma": [0, 0.1, 0.2]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=xgb_params,
    n_iter=10,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(x_train, y_train)

best_xgb = xgb_search.best_estimator_

xgb_pred = best_xgb.predict(x_val)

xgb_mse = mean_squared_error(y_val, xgb_pred)
xgb_mae = mean_absolute_error(y_val, xgb_pred)
xgb_r2 = r2_score(y_val, xgb_pred)

print("XGBoost Completed")

# ==========================================================
# CatBoost
# ==========================================================

print("\nTraining CatBoost...\n")

cat = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
    train_dir="model/catboost_info"
)
cat_params = {
    "iterations": [500, 800, 1000],
    "depth": [4, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05],
    "l2_leaf_reg": [1, 3, 5]
}

cat_search = RandomizedSearchCV(
    estimator=cat,
    param_distributions=cat_params,
    n_iter=10,
    cv=3,
    scoring="r2",
    random_state=42,
    n_jobs=-1
)

cat_search.fit(x_train, y_train)

best_cat = cat_search.best_estimator_

cat_pred = best_cat.predict(x_val)

cat_mse = mean_squared_error(y_val, cat_pred)
cat_mae = mean_absolute_error(y_val, cat_pred)
cat_r2 = r2_score(y_val, cat_pred)

print("CatBoost Completed")

# ==========================================================
# Metrics
# ==========================================================

metrics = {
    "Random Forest": {
        "MSE": float(rf_mse),
        "MAE": float(rf_mae),
        "R2": float(rf_r2)
    },
    "XGBoost": {
        "MSE": float(xgb_mse),
        "MAE": float(xgb_mae),
        "R2": float(xgb_r2)
    },
    "CatBoost": {
        "MSE": float(cat_mse),
        "MAE": float(cat_mae),
        "R2": float(cat_r2)
    }
}

# ==========================================================
# Best Model Selection
# ==========================================================

models = {
    "Random Forest": (best_rf, rf_search.best_params_, rf_r2),
    "XGBoost": (best_xgb, xgb_search.best_params_, xgb_r2),
    "CatBoost": (best_cat, cat_search.best_params_, cat_r2)
}

best_model_name = max(models, key=lambda x: models[x][2])

best_model = models[best_model_name][0]
best_params = models[best_model_name][1]

print("\nBest Model :", best_model_name)
print("Validation R2 :", models[best_model_name][2])

# ==========================================================
# Save Best Model
# ==========================================================

joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))

# ==========================================================
# Save Metrics
# ==========================================================

with open(os.path.join(evaluation_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# ==========================================================
# Save Best Parameters
# ==========================================================

best_params_dict = {
    "best_model": best_model_name,
    "best_params": best_params
}

with open(os.path.join(evaluation_dir, "best_params.json"), "w") as f:
    json.dump(best_params_dict, f, indent=4)

print("\nBest Model Saved Successfully")
print("Metrics Saved Successfully")
print("Best Parameters Saved Successfully")
print("\nModel Training Completed Successfully")