import os
import cv2
from typing import List, Any
from pydantic import BaseModel


class Image(BaseModel):
    image_name: str
    image_data: Any


class DataLoader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def load_data(self) -> List[Image]:
        files = self._get_files_name()
        images = []
        for file_name in files:
            image = cv2.imread(f"{self.directory_path}/{file_name}")
            images.append(Image(image_name=file_name, image_data=image))
        return images

    def _get_files_name(self) -> List[str]:
        file_names = []
        for file in os.listdir(self.directory_path):
            if os.path.isfile(os.path.join(self.directory_path, file)):
                file_names.append(file)
        return file_names
