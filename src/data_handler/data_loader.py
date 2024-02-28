import os
import cv2
import pandas as pd

from typing import List, Any
from pydantic import BaseModel
from typing import Literal


class Image(BaseModel):
    image_name: str
    data_type: str
    image_data: Any


class DataLoader:
    def __init__(self, directory_path: str, data_type: Literal["test", "train"]):
        self.directory_path = directory_path
        self.type = data_type

    def load_data(self) -> List[Image]:
        files = self._get_files_name()
        images = []
        for file_name in files:
            image = cv2.imread(f"{self.directory_path}/{file_name}")
            images.append(Image(image_name=file_name, image_data=image, data_type=self.type))
        return images

    def _get_files_name(self) -> List[str]:
        file_names = []
        for file in os.listdir(self.directory_path):
            if os.path.isfile(os.path.join(self.directory_path, file)):
                file_names.append(file)
        return file_names

    def get_labels(self, labels_path: str, images: List[Image]) -> List[str]:
        labels = []
        excel_file = labels_path
        df = pd.read_excel(excel_file)
        for image in images:
            for index, row in df.iterrows():
                if image.image_name == row["file_name"]:
                    labels.append(row['Label'])

        return labels
