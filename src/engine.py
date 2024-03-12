import os

from typing import Tuple, Literal
from src.models.vgg16 import Vgg16
from src.models.vgg19 import Vgg19
from src.models.xception import XceptionModel
from src.data_handler.data_loader import DataLoader
from src.data_handler.data_splitter import DataSplitter
from src.data_handler.pre_proc import PreProcess

VGG16 = "vgg16"
VGG19 = "vgg19"
XCEPTION = "xception"


class Engine:

    def __init__(self, image_shape: Tuple):
        self.model = None
        self.image_shape = image_shape
        self.train_data_path = None
        self.test_data_path = None
        self.train_labels = None
        self.test_labels = None
        self.train_images = None
        self.test_images = None

    def set_train_labels(self, df):
        self.train_labels = df

    def set_test_labels(self, df):
        self.test_labels = df

    def set_train_data_path(self, path: str):
        self.train_data_path = path

    def set_test_data_path(self, path: str):
        self.test_data_path = path

    def choose_model(self, model: str):
        if model == VGG16:
            self.model = Vgg16(input_shape=self.image_shape)
        elif model == VGG19:
            self.model = Vgg19(input_shape=self.image_shape)
        elif model == XCEPTION:
            self.model = XceptionModel(input_shape=self.image_shape)
        else:
            print(f"No model found with the name={model}")
            raise KeyError

    def load_images(self, data_type: Literal["test", "train"], name_col: str, label_col: str):
        data_path, dataframe = "", ""

        if data_type == "test":
            data_path = self.test_data_path
            dataframe = self.test_labels
        elif data_type == "train":
            data_path = self.train_data_path
            dataframe = self.train_labels

        data_loader = DataLoader(dataframe, data_type, data_path, name_col, label_col)
        images, labels = data_loader.load_data()

        # Preprocessing images
        preproc = PreProcess(self.image_shape)
        proc_images = preproc.resize_images(images)
        proc_labels = preproc.arrange_labels_indexing_from_0(labels)

        if data_type == "train":
            self.train_images, self.train_labels = proc_images, proc_labels
        elif data_type == "test":
            self.test_images, self.test_labels = proc_images, proc_labels

    def run_model(self):
        print("Running model...")
        self.model.run_model(self.train_images, self.train_labels)
        print("Evaluating model...")
        self.model.evaluation(self.test_images, self.test_labels)


if __name__ == "__main__":
    csv_label_path = "data\\AgeSplit.csv"
    ds = DataSplitter(csv_label_path)  # returns object with 'train', 'test', 'val' attributes

    image_shape = (400, 400, 3)
    engine = Engine(image_shape)

    engine.set_train_labels(ds.train)
    engine.set_test_labels(ds.test)

    engine.set_test_data_path(f"{os.getcwd()}\\data\\test")
    engine.set_train_data_path(f"{os.getcwd()}\\data\\train")

    print("Started loading images...")
    engine.load_images('train', 'File', 'Age')
    engine.load_images('test', 'File', 'Age')

    print("Finished loading images...")
    model = XCEPTION
    engine.choose_model(model)

    print(f"Chosen model: {model}")
    engine.run_model()
