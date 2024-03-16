import os
import pandas as pd
from pathlib import Path
import cv2
from typing import List, Any
from pydantic import BaseModel
from typing import Literal
from tqdm import tqdm


class Image(BaseModel):
    image_name: str
    data_type: str
    image_data: Any


class DataLoader:
    def __init__(self, dataframe, data_type: Literal["test", "train"], data_path: str, name_col: str, label_col: str):
        self.df = dataframe
        self.type = data_type
        self.path = data_path
        self.name_col = name_col
        self.label_col = label_col

    def load_data(self):

        files = self._get_files_name()
        images = []
        labels = []

        #@TODO : Experimentation should be done to find the best part size and overlap
        part_size = 256
        # Overlap - it is recommended to have a 25%-30% overlap from the part size
        overlap = 32

        for file_name in tqdm(files, total=len(files)):
            image = cv2.imread(str(Path(self.path) / file_name))

            for y in range(0, image.shape[0], part_size - overlap):
                for x in range(0, image.shape[1], part_size - overlap):
                    # Extract the current part
                    image_segment = image[y:y + part_size, x:x + part_size]

                    im = Image(image_name=f"{file_name}_{y}_{x}", image_data=image_segment, data_type=self.type)
                    lbl = self.df[self.df[self.name_col] == file_name][self.label_col].values[0]

                    images.append(im)
                    labels.append(lbl)

        return images, labels

    def _get_files_name(self) -> List[str]:
        file_names = []
        for file in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, file)):
                file_names.append(file)
        return file_names