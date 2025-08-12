from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import joblib
import librosa
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

# Root - show frontend
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        contents = await file.read()
        with open("temp_audio.wav", "wb") as f:
            f.write(contents)

        # Extract features
        y, sr = librosa.load("temp_audio.wav", duration=30)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        features = mfccs.reshape(1, -1)

        # Predict maqam
        prediction = model.predict(features)[0]

        return {"result": prediction}

    except Exception as e:
        return {"error": str(e)}
