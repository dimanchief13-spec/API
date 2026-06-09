from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from PIL import Image

from tensorflow import keras

import numpy as np
from PIL import Image
import io
import json
from pathlib import Path


app = FastAPI(
    title="Image Classification API",
    description="API для классификации изображений",
    version="1.0"
)

# Разрешаем запросы с фронтенда Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_classification_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

model = keras.models.load_model(MODEL_PATH, compile=False)
print(model.summary())

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)


def preprocess_image(image: Image.Image) -> np.ndarray:
    
    image = image.convert("RGB")
    image = image.resize((299, 299))
    image_array = np.array(image, dtype=np.float32)        # (299, 299, 3)
    image_array = keras.applications.inception_v3.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)      # (1, 299, 299, 3)
    return image_array


@app.get("/")
async def root():
    return {
        "message": "Image Classification API is running",
        "classes": class_names
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_input_shape": model.input_shape,
        "model_output_shape": model.output_shape,
        "classes": class_names
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image, verbose=0)[0]

        predicted_class_index = int(np.argmax(predictions))
        predicted_class_name = class_names[predicted_class_index]
        confidence = float(predictions[predicted_class_index])

        probabilities = {
            class_names[i]: float(predictions[i])
            for i in range(len(class_names))
        }

        return JSONResponse(content={
            "predicted_class": predicted_class_index,
            "predicted_class_name": predicted_class_name,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )