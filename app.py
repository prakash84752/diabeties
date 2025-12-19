from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("diabeties (1).pkl", "rb") as f:
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

