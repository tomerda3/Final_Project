from typing import Tuple, Literal

from src.models.vgg16 import VGG16Model
from src.models.vgg19 import VGG19Model
from src.models.xception import XceptionModel
from src.models.resnet import ResNet50Model
from src.models.efficientnet import EfficientNetModel
from src.data_handler.data_loader import DataLoader
from src.data_handler.label_splitter import *
from src.data_handler.pre_proc import PreProcess
from models.model_names import *
from collections import Counter
from sklearn.metrics import accuracy_score


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
            self.model = VGG16Model(input_shape=self.image_shape)
        elif model == VGG19:
            self.model = VGG19Model(input_shape=self.image_shape)
        elif model == Xception:
            self.model = XceptionModel(input_shape=self.image_shape)
        elif model == ResNet:
            self.model = ResNet50Model(input_shape=self.image_shape)
        elif model == EfficientNet:
            self.model = EfficientNetModel(input_shape=self.image_shape)
        else:
            print(f"No model found with the name={model}")
            raise KeyError

    def preprocess_data(self, images, labels, data_type):
        print(f"\nPreprocessing {data_type} images...")

        preprocessor = PreProcess(self.image_shape)

        proc_labels = preprocessor.arrange_labels_indexing_from_0(labels)
        reverse_binarize_images = preprocessor.grayscale_and_binarize_images(images)
        cropped_images = preprocessor.crop_text_from_reversed_binary_images(reverse_binarize_images)

        proc_images = cropped_images
        if data_type == "train":
            proc_images, proc_labels = preprocessor.patch_images(cropped_images, proc_labels, self.image_shape)
        # proc_images = preprocessor.resize_images(proc_images)

        return proc_images, proc_labels

    def load_images(self, data_type: Literal["test", "train"], image_filename_col: str, label_col: str,
                    clean_method: Literal["HHD", "KHATT"]="HHD"):

        data_path, dataframe = "", ""

        print(f"Loading {data_type} images...")
        if data_type == "test":
            data_path = self.test_data_path
            dataframe = self.test_labels
        elif data_type == "train":
            data_path = self.train_data_path
            dataframe = self.train_labels

        data_loader = DataLoader(dataframe, data_type, data_path, image_filename_col, label_col)
        images, labels = data_loader.load_data(clean_method)

        # Preprocessing:
        proc_images, proc_labels = self.preprocess_data(images, labels, data_type)

        if data_type == "train":
            self.train_images, self.train_labels = proc_images, proc_labels
        elif data_type == "test":
            self.test_images, self.test_labels = proc_images, proc_labels

    def train_model(self):
        print("Training model...")
        self.model.train_model(self.train_images, self.train_labels)

    def most_common_number(self, number_list):
        number_counts = Counter(number_list)
        most_common = number_counts.most_common(1)[0][0]
        return most_common

    def test_model(self):
        print("Evaluating model...")

        predictions = []
        preprocessor = PreProcess(self.image_shape)

        for i in range(len(self.test_images)):
            image = self.test_images[i]
            label = self.test_labels[i]

            patches, _ = preprocessor.patch_images([image], [label], self.image_shape)
            patch_predictions = self.model.patch_evaluation(patches)
            most_common_image_prediction = self.most_common_number(patch_predictions)
            predictions.append(most_common_image_prediction)

        accuracy = accuracy_score(self.test_labels, predictions)
        print(f"Accuracy: {accuracy}")
        print(f"Predictions: {predictions}")
        print(f"Real labels: {self.test_labels}")


def save_engine():
    # TODO: SAVE ENGINE TO DISC USING PICKLE / JOBLIB
    pass


def load_engine():
    # TODO: LOAD ENGINE FROM DISC USING PICKLE / JOBLIB
    pass


def construct_HHD_engine(base_dir, image_shape):

    # Setting file system
    train_path = base_dir / "train"
    test_path = base_dir / "test"
    csv_label_path = str(base_dir / "AgeSplit.csv")

    # Initializing engine
    engine = Engine(image_shape)

    # Setting engine labels & paths
    HHD_labels = LabelSplitter(csv_label_path)  # returns object with 'train', 'test', 'val' attributes
    engine.set_train_labels(HHD_labels.train)
    engine.set_test_labels(HHD_labels.test)
    engine.set_test_data_path(str(test_path))
    engine.set_train_data_path(str(train_path))

    engine.load_images(data_type='train', image_filename_col='File', label_col='Age')
    engine.load_images(data_type='test', image_filename_col='File', label_col='Age')

    return engine


def construct_KHATT_engine(base_dir, image_shape):

    # Setting file system
    train_path = base_dir / "Train"
    test_path = base_dir / "Test"
    csv_label_path = str(base_dir / "DatabaseStatistics-v1.0-NN.csv")

    # Initializing engine
    engine = Engine(image_shape)

    # Setting engine labels & paths
    KHATT_labels = LabelSplitter(csv_label_path, "Group", "R", "T", "V")  # gets 'train', 'test', 'val' attributes
    engine.set_train_labels(KHATT_labels.train)
    engine.set_test_labels(KHATT_labels.test)
    engine.set_test_data_path(str(test_path))
    engine.set_train_data_path(str(train_path))

    engine.load_images(data_type='train', image_filename_col='Form Number',
                       label_col='Age (1,2,3,or 4 from right to left)', clean_method="KHATT")
    engine.load_images(data_type='test', image_filename_col='Form Number',
                       label_col='Age (1,2,3,or 4 from right to left)', clean_method="KHATT")

    return engine
