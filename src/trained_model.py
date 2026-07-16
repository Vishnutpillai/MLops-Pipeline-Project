import os
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#file paths

feature_dir = "data/feature"
model_dir = "model"
evaluation_dir = "model/evaluation"

xtrain_path = os.path.join(feature_dir, "X_train.csv")
ytrain_path = os.path.join(feature_dir, "y_train.csv")

xtest_path = os.path.join(feature_dir, "X_test.csv")
ytest_path = os.path.join(feature_dir, "y_test.csv")

xval_path = os.path.join(feature_dir, "X_val.csv")
yval_path = os.path.join(feature_dir, "y_val.csv")

#create output directories

os.makedirs(model_dir, exist_ok=True)
os.makedirs(evaluation_dir, exist_ok=True)

#load dataset

print("="*50)
print("Loading dataset...")
print("="*50)

x_train = pd.read_csv(xtrain_path)
y_train = pd.read_csv(ytrain_path).squeeze() 

x_test = pd.read_csv(xtest_path)
y_test = pd.read_csv(ytest_path).squeeze()

x_val = pd.read_csv(xval_path)
y_val = pd.read_csv(yval_path).squeeze()

print("training shape:", x_train.shape)
print("test shape:", x_test.shape)
print("validation shape:", x_val.shape)

#Random forest model training

rf=RandomForestRegressor(random_state=42)

rf_params = {
    'n_estimators': [50,100,150],
    'max_depth': [10, 20,None],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(estimator=rf, param_grid=rf_params, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(x_train, y_train)

best_rf = rf_grid.best_estimator_

rf_pred=best_rf.predict(x_val)

rf_mse = mean_squared_error(y_val, rf_pred)
rf_mae = mean_absolute_error(y_val, rf_pred)
rf_r2 = r2_score(y_val, rf_pred)

print("Random forest completed")

#Xgboost model training

xgb=XGBRegressor(random_state=42,objective='reg:squarederror')

xgb_params = {
    'n_estimators': [50,100,150],   
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_grid = GridSearchCV(estimator=xgb, param_grid=xgb_params, cv=5, scoring='r2', n_jobs=-1)
xgb_grid.fit(x_train, y_train)

best_xgb = xgb_grid.best_estimator_

xgb_pred = best_xgb.predict(x_val)

xgb_mse = mean_squared_error(y_val, xgb_pred)
xgb_mae = mean_absolute_error(y_val, xgb_pred)
xgb_r2 = r2_score(y_val, xgb_pred)

print("Xgboost completed")

# Compare models

metrics = {
    "Random Forest": {"MSE": float(rf_mse), "MAE": float(rf_mae), "R2": float(rf_r2)},
    "XGBoost": {"MSE": float(xgb_mse), "MAE": float(xgb_mae), "R2": float(xgb_r2)}
}

if rf_r2 > xgb_r2:
    best_model = best_rf
    best_model_name = "Random Forest"
    best_params = rf_grid.best_params_

else:
    best_model = best_xgb
    best_model_name = "XGBoost"
    best_params = xgb_grid.best_params_

print("best model:", best_model_name)

#save the model

model_path = os.path.join(model_dir,"best_model.pkl")
joblib.dump(best_model, model_path)
print("best model saved")

#save the metrics

metrics_path = os.path.join(evaluation_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)
print("metrics saved")

#save the best model parameters

best_params_dict={
    "best_model": best_model_name,
    "best_params": best_params
}
params_path = os.path.join(evaluation_dir, "best_params.json")
with open(params_path, "w") as f:
    json.dump(best_params_dict, f, indent=4)
print("best model parameters saved")

print("Model Training Completed Successfully.")