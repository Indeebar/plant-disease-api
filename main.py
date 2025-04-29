import subprocess
import sys
import os
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Auto-install packages if missing
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import gdown
except ImportError:
    install('gdown')
    import gdown

try:
    import tensorflow as tf
except ImportError:
    install('tensorflow')
    import tensorflow as tf

# Download model from Google Drive if not already present
file_id = '10oBFt7djLQUTlzYyRHOJ_uia3c1yBkGZ'
gdrive_url = f"https://drive.google.com/uc?id={file_id}"
model_path = 'plant_disease_model.h5'

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(gdrive_url, model_path, quiet=False)
else:
    print("Model already exists. Skipping download.")

# Load the model
print("Loading model...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Load class labels
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
idx_to_class = {int(k): v for k, v in class_indices.items()}

# Create FastAPI app
app = FastAPI()

# ✅ Preprocessing to 128x128 (as per model training)
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ Root route
@app.get("/")
def read_root():
    return {"message": "🌿 Plant Disease Prediction API is running!"}

# ✅ Add HEAD route for Render health check
@app.head("/")
def head_root():
    return JSONResponse(content={}, status_code=200)

# ✅ Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_tensor = preprocess_image(contents)
        preds = model.predict(input_tensor)
        pred_class = np.argmax(preds, axis=1)[0]
        pred_label = idx_to_class.get(pred_class, "Unknown")
        confidence = float(np.max(preds))

        return {
            "predicted_class": pred_label,
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
