import os
import logging
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Configure Logging
logging.basicConfig(
    filename="logs/api_requests.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths to Model, Scaler, and Feature Names
MODEL_PATH = "models/fraud_RandomForest.pkl"
SCALER_PATH = "models/fraud_scaler.pkl"
FEATURE_NAMES_PATH = "models/fraud_feature_names.pkl"

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load Model, Scaler, and Feature Names
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    logging.info("Model, Scaler, and Feature Names loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or dependencies: {str(e)}")
    model, scaler, feature_names = None, None, None

@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for fraud detection"""
    try:
        # Get JSON data
        data = request.get_json()

        if not isinstance(data, dict):
            raise ValueError("Input data should be a dictionary.")

        df = pd.DataFrame([data])

        # Ensure all expected features are present
        missing_features = [col for col in feature_names if col not in df.columns]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Reorder columns to match the trained model
        df = df[feature_names]

        # Scale numerical features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = scaler.transform(df[numeric_columns])

        # Make prediction
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)[:, 1]  # Probability of fraud

        # Logging request and response
        logging.info(f"Received request: {data}")
        logging.info(f"Prediction: {int(prediction[0])}, Probability: {prediction_proba[0]:.4f}")

        return jsonify({
            "fraud_prediction": int(prediction[0]),
            "fraud_probability": round(prediction_proba[0], 4)
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
