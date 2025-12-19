from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Get absolute path of current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct model file name (NO spaces, NO brackets)
MODEL_PATH = os.path.join(BASE_DIR, "diabetes.pkl")

# Load the trained model safely
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Diabetes Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting input features as a list
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

