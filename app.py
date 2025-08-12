import pickle
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn

MODEL_PATH = "model.pkl"

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI()

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

@app.get("/")
def root():
    return {"message": "Arabic Maqam Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save temp file
    with open("temp_audio", "wb") as f:
        f.write(await file.read())

    # Extract features & predict
    features = extract_features("temp_audio").reshape(1, -1)
    prediction = model.predict(features)[0]

    return {"maqam": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
