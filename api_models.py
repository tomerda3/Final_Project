import base64
from pydantic import BaseModel
from typing import Literal


class ModelRequest(BaseModel):
    model_name: str
    image: base64
    data_set: Literal['HDD', 'KHATT']


class ModelResults(BaseModel):
    model_name: str
    data_set: str
    predictions: str
