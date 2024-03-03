from typing import Tuple, Literal
from src.models.vgg16 import Vgg16
from src.models.vgg19 import Vgg19
from src.models.xception import XceptionModel
from src.data_handler.data_loader import DataLoader
from src.data_handler.data_splitter import DataSplitter


VGG16 = "vgg16"
VGG19 = "vgg19"
XCEPTION = "xception"


class Engine:

    def __init__(self, image_shape: Tuple):
        self.image_shape = image_shape
        self.train_data_path = None
        self.test_data_path = None
        self.train_labels = None
        self.test_labels = None

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

    def _load_images(self, dir_path: str, data_type: Literal["test", "train"], labels_path: str):
        data_loader = DataLoader(dir_path, data_type)
        data = data_loader.load_data()
        labels = data_loader.get_labels(labels_path, data)
        return data, labels

    def run_model(self):
        self.model.run_model(self.train_images, self.train_labels)
        self.model.evaluation(self.test_images, self.test_labels)


if __name__ == "__main__":
    image_shape = (400, 400)

    csv_label_path = "data\\AgeSplit.csv"
    ds = DataSplitter(csv_label_path)

    engine = Engine(image_shape)

    engine.set_train_labels(ds.train)
    engine.set_test_labels(ds.test)

    engine.choose_model("vgg16")

    # engine.run_model()

