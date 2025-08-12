from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
import librosa
import numpy as np

app = FastAPI()

# ✅ Allow requests from anywhere (or specify your site)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["file://", "http://localhost", "https://your-site.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"message": "Arabic Maqam Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    y, sr = librosa.load(file.file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfcc, axis=1).reshape(1, -1)
    prediction = model.predict(features)
    return {"maqam": prediction[0]}
