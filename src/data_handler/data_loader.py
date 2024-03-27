import os
import pandas as pd
from pathlib import Path
import cv2
from typing import List
from typing import Literal
from tqdm import tqdm

SHORT_RUN = True
IMAGE_LIMIT = 10


class DataLoader:
    def __init__(self, dataframe, data_type: Literal["test", "train"], data_path: str, name_col: str, label_col: str):
        self.df = dataframe
        self.type = data_type
        self.path = data_path
        self.name_col = name_col
        self.label_col = label_col

    def load_data(self, clean_method: Literal["HHD", "KHATT"]="HHD"):

        files = self._get_files_name()
        images = []
        labels = []

        if SHORT_RUN:
            cnt = 0

        for file_name in tqdm(files):

            if SHORT_RUN:
                cnt += 1
                if cnt == IMAGE_LIMIT:
                    return images, labels

            image = cv2.imread(str(Path(self.path) / file_name))  # TODO: Read as greyscale -> remove from preprocess

            if clean_method == "KHATT":
                clean_name = file_name[5:10]
                row_of_file = self.df[self.df[self.name_col] == clean_name]
                if len(row_of_file) == 0:  # if filename not in labels database
                    continue
                lbl = row_of_file[self.label_col].values[0]
            else:
                lbl = self.df[self.df[self.name_col] == file_name][self.label_col].values[0]

            images.append(image)
            labels.append(lbl)

        return images, labels

    def _get_files_name(self) -> List[str]:
        file_names = []
        for file in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, file)):
                file_names.append(file)
        return file_names
