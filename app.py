from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import librosa
import numpy as np
import uvicorn
import os
import shutil

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# -----------------------------
# Create app
# -----------------------------
app = FastAPI()

# Allow CORS (optional since HTML will be served from same domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Serve HTML
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    return FileResponse("index.html")

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract features
        y, sr = librosa.load(temp_file, mono=True, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

        # Predict
        prediction = model.predict(mfcc_mean)[0]

        # Cleanup
        os.remove(temp_file)

        return JSONResponse({"maqam": str(prediction)})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
# Local run
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
