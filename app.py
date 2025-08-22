from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import os
from preprocess import preprocess_csv
from supabase_logger import log_prediction


# Load model
MODEL_PATH = "schizophrenia_detection_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# API key from environment
API_KEY = os.getenv("API_KEY", "my-secret-key")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["https://lovable.ai"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    contents = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        eeg_data = preprocess_csv(contents)
    elif filename.endswith(".edf"):
        eeg_data = preprocess_edf(contents)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    eeg_array = np.array(eeg_data, dtype=float)[:252].reshape(1, 252, 1)
    y_pred = model.predict(eeg_array)
    predicted_class = int(np.argmax(y_pred))
    probabilities = y_pred.tolist()

    log_prediction(filename, eeg_array.tolist(), predicted_class, probabilities)

    return {
        "filename": filename,
        "class": predicted_class,
        "probabilities": probabilities
    }
