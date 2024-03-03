from typing import Tuple, Literal
from src.models.vgg16 import Vgg16
from src.models.vgg19 import Vgg19
from src.models.xception import XceptionModel
from src.data_handler.data_loader import DataLoader


VGG16 = "vgg16"
VGG19 = "vgg19"
XCEPTION = "xception"


class Engine:

    def __init__(self, image_shape: Tuple, train_data_path, test_data_path, train_labels_path, test_labels_path):
        self.image_shape = image_shape
        self.train_image, self.train_labels = self._load_images(train_data_path, "train", train_labels_path)
        self.test_image, self.test_labels = self._load_images(test_data_path, "test", test_labels_path)

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
        self.model.run_model(self.train_image, self.train_labels)
        self.model.evaluation(self.test_image, self.test_labels)


if __name__ == "__main__":
    image_shape = (400, 400)

    train = "//...//...//"
    test = "//...//...//"

    train_labels = "//...//...//.xls"
    test_labels = "//...//...//.xls"

    engine = Engine(image_shape, train, test, train_labels, test_labels)
    engine.choose_model("vgg16")

    engine.run_model()
