from typing import Tuple, Literal
from src.models.vgg16 import Vgg16
from src.models.vgg19 import Vgg19
from src.models.xception import XceptionModel
from src.data_handler.data_loader import DataLoader
from src.data_handler.label_splitter import LabelSplitter
from src.data_handler.pre_proc import PreProcess
from models.model_names import *


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

    def preprocess_data(self, images, labels, data_type):
        print(f"\nPreprocessing {data_type} images...")

        preprocessor = PreProcess(self.image_shape)
        proc_labels = preprocessor.arrange_labels_indexing_from_0(labels)
        grayscale_images = preprocessor.grayscale_images(images)
        reverse_binarize_images = preprocessor.reverse_binarize_images(grayscale_images)
        cropped_images = preprocessor.crop_text_from_reversed_binary_images(reverse_binarize_images)
        # proc_images, proc_labels = preprocessor.segment_images(cropped_images, proc_labels, self.image_shape)
        proc_images = preprocessor.resize_images(cropped_images)  # TODO: replace with segmentation

        return proc_images, proc_labels

    def load_images(self, data_type: Literal["test", "train"], image_filename_col: str, label_col: str):
        data_path, dataframe = "", ""

        print(f"Loading {data_type} images...")
        if data_type == "test":
            data_path = self.test_data_path
            dataframe = self.test_labels
        elif data_type == "train":
            data_path = self.train_data_path
            dataframe = self.train_labels

        data_loader = DataLoader(dataframe, data_type, data_path, image_filename_col, label_col)
        images, labels = data_loader.load_data()

        # Preprocessing:
        proc_images, proc_labels = self.preprocess_data(images, labels, data_type)

        if data_type == "train":
            self.train_images, self.train_labels = proc_images, proc_labels
        elif data_type == "test":
            self.test_images, self.test_labels = proc_images, proc_labels

    def train_model(self):
        print("Training model...")
        self.model.train_model(self.train_images, self.train_labels)

    def test_model(self):
        print("Evaluating model...")
        self.model.evaluation(self.test_images, self.test_labels)

    def save_model(self):
        # TODO: SAVE MODEL TO DISC USING PICKLE
        pass

    def load_model(self):
        # TODO: LOAD MODEL FROM DISC USING PICKLE
        pass


def get_bundled_engine(base_dir, train_images_folder, test_images_folder, labels_file, image_shape):

    # Setting file system
    train_path = base_dir / train_images_folder
    test_path = base_dir / test_images_folder
    csv_label_path = str(base_dir / labels_file)

    # Initializing engine
    engine = Engine(image_shape)

    # Setting engine labels & paths
    get_labels = LabelSplitter(csv_label_path)  # returns object with 'train', 'test', 'val' attributes
    engine.set_train_labels(get_labels.train)
    engine.set_test_labels(get_labels.test)
    engine.set_test_data_path(str(test_path))
    engine.set_train_data_path(str(train_path))

    return engine
