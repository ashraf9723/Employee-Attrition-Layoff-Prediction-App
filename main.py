import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
from  skopt import BayesSearchCV  # Ensure scikit-optimize is installed with: pip install scikit-optimize

# Load datasets
attrition_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
layoff_data = pd.read_csv('layoffs_data.csv')

# Data Preprocessing for Attrition Prediction
le = LabelEncoder()
for col in attrition_data.select_dtypes(include=['object']).columns:
    attrition_data[col] = le.fit_transform(attrition_data[col])

X_attrition = attrition_data.drop(columns=['Attrition'])
y_attrition = attrition_data['Attrition']
X_train_attr, X_test_attr, y_train_attr, y_test_attr = train_test_split(
    X_attrition, y_attrition, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_attr = scaler.fit_transform(X_train_attr)
X_test_attr = scaler.transform(X_test_attr)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_attr, y_train_attr)
y_pred_attr = rf_classifier.predict(X_test_attr)
print(f'Attrition Prediction Accuracy: {accuracy_score(y_test_attr, y_pred_attr) * 100:.2f}%')

# Save the attrition model and scaler
joblib.dump(rf_classifier, 'attrition_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


# Load the layoffs dataset
file_path = "layoffs_data.csv"  # Update with actual file path
layoff_data = pd.read_csv(file_path)

# Drop unnecessary columns
layoff_data = layoff_data.drop(columns=["Date", "Source", "List_of_Employees_Laid_Off", "Date_Added"])

# Fill missing values
layoff_data["Funds_Raised"].fillna(layoff_data["Funds_Raised"].median(), inplace=True)
layoff_data["Percentage"].fillna(layoff_data["Percentage"].median(), inplace=True)
layoff_data.dropna(subset=["Laid_Off_Count"], inplace=True)  # Remove rows where target is missing

# Encode categorical columns
le = LabelEncoder()
for col in ["Company", "Location_HQ", "Industry", "Stage", "Country"]:
    layoff_data[col] = le.fit_transform(layoff_data[col])

# Define features and target
X_layoff = layoff_data.drop(columns=["Laid_Off_Count"])
y_layoff = layoff_data["Laid_Off_Count"]

# Split dataset
X_train_lay, X_test_lay, y_train_lay, y_test_lay = train_test_split(
    X_layoff, y_layoff, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_lay = scaler.fit_transform(X_train_lay)
X_test_lay = scaler.transform(X_test_lay)

# Train an initial XGBoost model for feature importance analysis
xgb_temp = XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_temp.fit(X_train_lay, y_train_lay)

# Feature importance analysis
feature_importances = pd.Series(xgb_temp.feature_importances_, index=X_layoff.columns)
important_features = feature_importances[feature_importances > 0.01].index  # Keep only important features
X_train_lay = pd.DataFrame(X_train_lay, columns=X_layoff.columns)[important_features]
X_test_lay = pd.DataFrame(X_test_lay, columns=X_layoff.columns)[important_features]

# Bayesian Optimization for XGBoost
param_grid = {
    'n_estimators': (100, 500),
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 10),
    'subsample': (0.7, 1.0)
}

xgb_regressor = XGBRegressor(objective='reg:squarederror', random_state=42)
opt = BayesSearchCV(xgb_regressor, param_grid, n_iter=20, cv=3, n_jobs=-1, random_state=42)
opt.fit(X_train_lay, y_train_lay)

# Best model after optimization
best_xgb_regressor = opt.best_estimator_
y_pred_xgb = best_xgb_regressor.predict(X_test_lay)
xgb_mae = mean_absolute_error(y_test_lay, y_pred_xgb)
print(f'Optimized XGBoost MAE: {xgb_mae:.4f}')

# Save the best model and scaler
joblib.dump(best_xgb_regressor, 'optimized_xgboost_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
