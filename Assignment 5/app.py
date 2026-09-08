# ==============================================================================
# Crop Recommendation System - Flask REST API
# B.Tech 7th Sem - Advanced Data Science Laboratory
# ==============================================================================

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained Random Forest model
MODEL_PATH = "model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("Warning: model.pkl not found! Please run train_model.py first.")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "Crop Recommendation System API",
        "status": "running",
        "endpoints": {
            "predict": "POST /predict"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model not loaded. Please train and save model.pkl first."
        }), 500

    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON input or Content-Type not set to application/json."
        }), 400

    # Extract required features (supports both 'ph' and 'pH')
    try:
        n = float(data.get("N"))
        p = float(data.get("P"))
        k = float(data.get("K"))
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        ph = float(data.get("ph") if "ph" in data else data.get("pH"))
        rainfall = float(data.get("rainfall"))
    except (TypeError, ValueError):
        return jsonify({
            "status": "error",
            "message": "All 7 features (N, P, K, temperature, humidity, ph, rainfall) must be provided as numbers."
        }), 400

    # Create DataFrame with exact feature names matching the training set
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_df = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]], columns=feature_names)
    
    # Perform prediction
    prediction = model.predict(input_df)[0]

    return jsonify({
        "status": "success",
        "recommended_crop": prediction,
        "input_features": {
            "N": n,
            "P": p,
            "K": k,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
