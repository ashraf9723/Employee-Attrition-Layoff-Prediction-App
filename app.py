import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import Flask, request, jsonify, render_template
import joblib
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Load the trained models and scaler
layoff_model_path = "optimized_xgboost_model.pkl"
attrition_model_path = "attrition_model.pkl"
scaler_path = "scaler.pkl"
if os.path.exists(layoff_model_path) and os.path.exists(attrition_model_path) and os.path.exists(scaler_path):
    layoff_model = joblib.load(layoff_model_path)
    attrition_model = joblib.load(attrition_model_path)
    scaler = joblib.load(scaler_path)
else:
    raise FileNotFoundError("One or more model files not found!")

# Initialize Flask app
app = Flask(__name__)

# Define categorical columns for encoding
categorical_columns = ["Company", "Location_HQ", "Industry", "Stage", "Country", "Department", "Job_Role"]


def preprocess_input(data):
    """Preprocess user input data for both attrition and layoff predictions"""
    df = pd.DataFrame([data])

    # Encode categorical features
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Scale numerical features
    df_scaled = scaler.transform(df)
    return df_scaled


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        processed_data = preprocess_input(data)
        layoff_prediction = layoff_model.predict(processed_data)[0]
        attrition_prediction = attrition_model.predict_proba(processed_data)[0][1] * 100  # Probability of attrition
        return jsonify({
            "layoff_prediction": round(layoff_prediction, 2),
            "attrition_prediction": round(attrition_prediction, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
