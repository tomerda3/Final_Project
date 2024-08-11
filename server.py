import PIL
import base64
import uvicorn
from PIL import Image
from typing import List
from typing import Dict
from typing import Any
from io import BytesIO
from fastapi import FastAPI
from api_models import ModelRequest
from backend_middleware import BackendMiddleware
from src.models.model_names import models_list

app = FastAPI()


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
    image_data = base64.b64decode(image)
    image = Image.open(BytesIO(image_data))
    return image


def get_results(data_set, model_name, image) -> Dict[str, Any]:
    middleware = BackendMiddleware()
    results = middleware.create_and_run_engine(data_set, model_name, image)
    return results.dict()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
