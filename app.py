from fastapi import FastAPI, UploadFile, File
import joblib
import librosa
import numpy as np
import uvicorn

# Load trained model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Arabic Maqam Classifier")

# Feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    # Extract features
    features = extract_features(file.filename)

    # Predict maqam
    prediction = model.predict(features)[0]

    return {"predicted_maqam": prediction}

@app.get("/")
def home():
    return {"message": "Arabic Maqam Classifier API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
