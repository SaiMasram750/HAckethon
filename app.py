# app.py

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import csv
from io import StringIO

app = Flask(__name__)

# Load the model once when the app starts
try:
    MODEL_PATH = "schizophrenia_detection_model.h5"
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if model fails to load, preventing a startup crash
    exit()

# Preprocessing function (same as before)
def preprocess_csv(contents):
    # Your CSV preprocessing logic here
    try:
        csv_data = contents.decode('utf-8')
        csv_file = StringIO(csv_data)
        reader = csv.reader(csv_file)
        rows = list(reader)[1:]
        processed_data = [float(val) for row in rows for val in val] # Example: adjust as needed
        return processed_data
    except Exception as e:
        raise ValueError(f"Error processing CSV: {e}")

# Add the same logic for preprocess_edf if needed
def preprocess_edf(contents):
    raise NotImplementedError("EDF processing is not implemented.")

# The API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Authentication check
    api_key = request.headers.get('X-Api-Key')
    if api_key != os.getenv("API_KEY"):
        return jsonify({"detail": "Invalid API Key"}), 403

    # Check for file in the request
    if 'file' not in request.files:
        return jsonify({"detail": "No file provided"}), 400

    file = request.files['file']
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        contents = file.read()
        try:
            eeg_data = preprocess_csv(contents)
        except ValueError as e:
            return jsonify({"detail": str(e)}), 400
    elif filename.endswith(".edf"):
        # Handle EDF file
        pass # Not implemented in this example
    else:
        return jsonify({"detail": "Unsupported file type"}), 400

    # Prediction logic
    eeg_array = np.array(eeg_data, dtype=float)[:252].reshape(1, 252, 1)
    y_pred = model.predict(eeg_array)
    predicted_class = int(np.argmax(y_pred))
    probabilities = y_pred.tolist()

    return jsonify({
        "filename": filename,
        "class": predicted_class,
        "probabilities": probabilities
    })

if __name__ == '__main__':
    # Flask will be run by a production server like Gunicorn, not this block.
    # This is for local testing only.
    app.run(host='0.0.0.0', port=5000)
