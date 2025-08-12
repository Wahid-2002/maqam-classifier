import os
import pickle
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Set your dataset path (each maqam should be a folder with audio files)
DATASET_PATH = "dataset"

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

X, y = [], []
maqams = os.listdir(DATASET_PATH)

for maqam in maqams:
    maqam_path = os.path.join(DATASET_PATH, maqam)
    if not os.path.isdir(maqam_path):
        continue
    for file in os.listdir(maqam_path):
        if file.endswith(".mp3") or file.endswith(".wav"):
            features = extract_features(os.path.join(maqam_path, file))
            X.append(features)
            y.append(maqam)

# Train model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("✅ model.pkl saved successfully.")
