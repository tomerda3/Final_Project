import PIL
import base64
import cv2
import numpy as np
import uvicorn
from PIL import Image
from typing import List
from typing import Dict
from typing import Any
from io import BytesIO
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api_models import ModelRequest
from backend_middleware import BackendMiddleware
from src.models.models_metadata import models_list

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def is_alive() -> str:
    return "Iam alive"


@app.get("/models")
def get_models_name() -> List[str]:
    return models_list


@app.post("/get_prediction")
def run_model(request: ModelRequest) -> Dict[str, Any]:
    image = open_image_from_b64(request.image)
    data_set = request.data_set
    model = request.model_name
    results = get_results(data_set, model, image)
    return results


def open_image_from_b64(image: base64) -> PIL.Image:
    if ',' in image:
        image = image.split(',')[1]
    image_data = base64.b64decode(image)
    image = Image.open(BytesIO(image_data))
    pillow_image = image.convert('RGB')
    image_array = np.array(pillow_image)
    print(image_array.shape)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    gray_image = image_bgr[:, :, np.newaxis]
    print(gray_image.shape)
    return image_bgr


def get_results(data_set, model_name, image) -> Dict[str, Any]:
    middleware = BackendMiddleware()
    results = middleware.create_and_run_engine(data_set, model_name, image)
    return results.dict()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
