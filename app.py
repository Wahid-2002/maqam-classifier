from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import librosa
import numpy as np
import uvicorn
import os, shutil

MODEL_PATH = "model.pkl"                # <- your existing file
model = joblib.load(MODEL_PATH)         # <- IMPORTANT: joblib.load

app = FastAPI(title="Arabic Maqam Classifier")

# (Optional but harmless) – allows browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("index.html")

def extract_features(path):
    y, sr = librosa.load(path, mono=True, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        tmp = f"tmp_{file.filename}"
        with open(tmp, "wb") as f:
            f.write(await file.read())

        feats = extract_features(tmp)
        pred = model.predict(feats)[0]
        os.remove(tmp)

        return {"maqam": str(pred)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
