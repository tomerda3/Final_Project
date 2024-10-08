import os
from pathlib import Path
import cv2
from typing import List
from typing import Literal
from tqdm import tqdm
from ..data.path_variables import *

SHORT_RUN = False
IMAGE_LIMIT = 15


class DataLoader:
    def __init__(self, dataframe, data_type: Literal["test", "train"], data_path: str, name_col: str, label_col: str):
        self.df = dataframe
        self.type = data_type
        self.path = data_path
        self.name_col = name_col
        self.label_col = label_col

    def load_data(self, clean_method: Literal[HHD, KHATT] = HHD):

        files = self._get_files_name()
        images = []
        labels = []
        filenames = []

        if SHORT_RUN:
            print("\nSHOT RUN IS SELECTED! (in data_loader.py)")
            files = files[:IMAGE_LIMIT]

        for file_name in tqdm(files):

            image = cv2.imread(str(Path(self.path) / file_name))  # , cv2.IMREAD_GRAYSCALE

            if clean_method == KHATT:
                clean_name = file_name[5:10]
                row_of_file = self.df[self.df[self.name_col] == clean_name]
                if len(row_of_file) == 0:  # if filename not in labels database
                    continue
                lbl = row_of_file[self.label_col].values[0]
            else:
                lbl = self.df[self.df[self.name_col] == file_name][self.label_col].values[0]

            images.append(image)
            labels.append(lbl)
            filenames.append(file_name)  # ADDED FOR REGRESSION
        return images, labels, filenames

    def _get_files_name(self) -> List[str]:
        file_names = []
        for file in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, file)):
                file_names.append(file)
        return file_names
