import os
from typing import Tuple, Literal
import cv2
import numpy as np
from models.vgg16 import VGG16Model
from models.vgg19 import VGG19Model
from models.xception import XceptionModel
from models.resnet50 import ResNet50Model
from models.efficientnet import EfficientNetModel
from models.efficientnetv2 import EfficientNetV2LModel
from models.mobilenet import MobileNetV2Model
from models.resnet152v2 import ResNet152V2Model
from models.convnextxl import ConvNeXtXLargeModel
from data_handler.data_loader import DataLoader
from data_handler.label_splitter import *
from data_handler.pre_proc import PreProcess
from models import model_names
from collections import Counter
from sklearn.metrics import accuracy_score
from confusion_matrix import ConfusionMatrixGenerator
#####   Tourch Libarys #####
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
from PIL import Image
#####   Tourch Libarys #####
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
        self.data_name = None
        self.model_name = None
        self.confusion_matrix_generator = ConfusionMatrixGenerator()

    def set_train_labels(self, df):
        self.train_labels = df

    def set_test_labels(self, df):
        self.test_labels = df

    def set_train_data_path(self, path: str):
        self.train_data_path = path

    def set_test_data_path(self, path: str):
        self.test_data_path = path

    def set_model(self, model: str):
        self.model_name = model

        if model == model_names.VGG16:
            self.model = VGG16Model(input_shape=self.image_shape)
        elif model == model_names.VGG19:
            self.model = VGG19Model(input_shape=self.image_shape)
        elif model == model_names.Xception:
            self.model = XceptionModel(input_shape=self.image_shape)
        elif model == model_names.ResNet50:
            self.model = ResNet50Model(input_shape=self.image_shape)
        elif model == model_names.EfficientNet:
            self.model = EfficientNetModel(input_shape=self.image_shape)
        elif model == model_names.MobileNet:
            self.model = MobileNetV2Model(input_shape=self.image_shape)
        elif model == model_names.ResNet152v2:
            self.model = ResNet152V2Model(input_shape=self.image_shape)
        elif model == model_names.EfficientNetV2:
            self.model = EfficientNetV2LModel(input_shape=self.image_shape)
        elif model == model_names.ConvNeXtXLarge:
            self.model = ConvNeXtXLargeModel(input_shape=self.image_shape)
        else:
            print(f"No model found with the name={model}")
            raise KeyError
    def get_pytorch_transforms(self,image_shape: Tuple[int, int]):
        image_shape = (image_shape[0], image_shape[1])
        return transforms.Compose([
            transforms.Resize(image_shape),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor (example values)
        ])

    def preprocess_data(self, images, labels, data_type):
        print(f"\nPreprocessing {data_type} images...")
        preprocessor = PreProcess(self.image_shape)
        if data_type == "train":
            images, labels = preprocessor.patch_images(images, labels, self.image_shape)
        transform = self.get_pytorch_transforms(self.image_shape)
        proc_images = [transform(Image.fromarray(img)) for img in images]
        proc_labels = [label - 1 for label in labels]  # Adjust label indexing if necessary
        proc_images = [img.permute(1, 2, 0) for img in proc_images]
        proc_images = [img.unsqueeze(-1) for img in proc_images]
        return torch.stack(proc_images), torch.tensor(proc_labels)

    def load_images(self, data_type: Literal["test", "train"], image_filename_col: str, label_col: str,
                    clean_method: Literal["HHD", "KHATT"]="HHD"):
        self.data_name = clean_method
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

    def save_run_txt(self, accuracy):
        model_name = self.model_name
        dataset_name = self.data_name
        res_dir = f"./run_results/{dataset_name}"

        # Ensure the directory exists, create if not
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        # Define the file path
        file_path = os.path.join(res_dir, f"{model_name}_results.txt")

        # Write the text to the file
        with open(file_path, 'w') as f:
            f.write(f"Accuracy: {accuracy}")

    def test_model(self):
        print("Evaluating model...")

        real_labels = []  # TODO: remove after fixing empty patches sequence
        predictions = []
        preprocessor = PreProcess(self.image_shape)

        for i in range(len(self.test_images)):
            image = self.test_images[i]
            label = self.test_labels[i]

            patches, _ = preprocessor.patch_images([image], [label], self.image_shape)
            if len(patches) == 0:  # TODO: FIX PATCHES SOMETIMES BEING EMPTY
                print(f"Image with shape {image.shape} skipped, real label {label}")
                continue
            patch_predictions = self.model.patch_evaluation(patches)
            most_common_image_prediction = self.most_common_number(patch_predictions)
            predictions.append(most_common_image_prediction)
            real_labels.append(label)  # TODO: remove after fixing empty patches sequence

        # accuracy = accuracy_score(self.test_labels, predictions)
        accuracy = accuracy_score(real_labels, predictions)
        print(f"Accuracy: {accuracy}")
        print(f"Predictions: {predictions}")
        # print(f"Real labels: {self.test_labels}")
        print(f"Real labels: {real_labels}")
        # self.confusion_matrix_generator.build_confusion_matrix_plot(self.test_labels,
        self.confusion_matrix_generator.build_confusion_matrix_plot(real_labels,
                                                                    predictions,
                                                                    self.model_name,
                                                                    self.data_name)
        self.save_run_txt(accuracy)



def save_engine():  # TODO: SAVE ENGINE TO DISC USING PICKLE / JOBLIB
    """
    After lots of trials and errors,
    I figured that the best direction to tackle this is to only save weights directly after training:
        model.save_weights()

    We can then load weights back using:
        model.load_weights()

    So far I have tried:
        Using Joblib instead of pickle,
        Not using Lambda layers manually,
        Save model using keras.saving.save_model() WHICH WORKED! Could not load the model because of Lambda layer
        The Lambda layer is apparently not removable because it is already in the base model layers.
    """
    pass


def load_engine():  # TODO: LOAD ENGINE FROM DISC USING PICKLE / JOBLIB
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
