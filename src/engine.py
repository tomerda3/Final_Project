import os
from pathlib import Path
from typing import Tuple, Literal
import numpy as np
import pandas as pd
from data.path_variables import *
import statistics

from src.models.vgg16 import VGG16Model
from src.models.vgg19 import VGG19Model
from src.models.xception import XceptionModel
from src.models.resnet50 import ResNet50Model
from src.models.efficientnet import EfficientNetModel
from src.models.efficientnetv2 import EfficientNetV2LModel
from src.models.mobilenet import MobileNetV2Model
from src.models.resnet152v2 import ResNet152V2Model
from src.models.convnextxl import ConvNeXtXLargeModel
from src.models.convnextxlregression import ConvNeXtXLargeRegressionModel
from src.data_handler.data_loader import DataLoader
from src.data_handler.label_splitter import *
from src.data_handler.pre_proc import PreProcess
from src.models import model_names
from collections import Counter
from sklearn.metrics import accuracy_score, mean_absolute_error
from src.confusion_matrix import ConfusionMatrixGenerator


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
        self.is_regression = False
        self.train_filenames = None
        self.test_filenames = None
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
        elif model == model_names.ConvNeXtXLargeRegression:
            self.model = ConvNeXtXLargeRegressionModel(input_shape=self.image_shape)
            if self.data_name != HHD:
                print("Regression only works with HHD dataset.")
                exit(0)
            self.is_regression = True
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
            proc_images, proc_labels = preprocessor.patch_images(proc_images, proc_labels, self.image_shape)

        return proc_images, proc_labels

    def load_images(self, data_type: Literal["test", "train"], image_filename_col: str, label_col: str,
                    clean_method: Literal[HHD, KHATT] = HHD):
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
        images, labels, filenames = data_loader.load_data(clean_method)
        # Preprocessing:
        proc_images, proc_labels = self.preprocess_data(images, labels, data_type)

        if data_type == "test":
            self.test_filenames = filenames
        elif data_type == "train":
            self.train_filenames = filenames

        if data_type == "train":
            self.train_images, self.train_labels = proc_images, proc_labels
        elif data_type == "test":
            self.test_images, self.test_labels = proc_images, proc_labels

    def train_model(self):
        if self.is_regression:
            self.train_filenames, self.train_labels, self.train_images = self.get_regression_values(
                self.train_filenames, self.train_labels, self.train_images)

        print("Training model...")
        self.model.train_model(self.train_images, self.train_labels, self.data_name)

    def get_regression_values(self, filenames, labels, images):
        name_mapping = pd.read_csv(Path.cwd() / DATA / HHD / "NameMapping.csv", header=None)
        real_ages = pd.read_csv(Path.cwd() / DATA / HHD / "RealAges.csv")
        print("SETTING UP REAL AGES FOR REGRESSION MODEL")
        # Initialize the lists
        keep_indices = []
        for i in range(len(filenames)):
            file_name = filenames[i]
            name_in_real_ages_data = name_mapping[name_mapping[1] == file_name][0].values[0][:-4] + ".jpg"

            real_age = real_ages[real_ages['name'] == name_in_real_ages_data]['age'].values
            if len(real_age) == 0:
                # Skip this index if real_age is not found
                continue
            keep_indices.append(i)

            # Set the label to the correct age
            real_age = real_age[0]
            labels[i] = real_age
        # Convert list to numpy array for indexing
        keep_indices = np.array(keep_indices)
        # Apply mask to filter out the valid entries
        try:
            images = images[keep_indices]
        except:
            images = [images[i] for i in keep_indices]
        labels = [labels[i] for i in keep_indices]
        filenames = [filenames[i] for i in keep_indices]

        return filenames, labels, images

    # def most_common_number(self, number_list):
    #     number_counts = Counter(number_list)
    #     most_common = number_counts.most_common(1)[0][0]
    #     return most_common

    def most_common_number(self, number_list):
        # Convert each numpy array in the list to a tuple
        hashable_number_list = [tuple(num) if isinstance(num, np.ndarray) else num for num in number_list]
        number_counts = Counter(hashable_number_list)
        most_common = number_counts.most_common(1)[0][0]  # Get the most common element
        return most_common

    def save_run_txt(self, accuracy, predictions, real_labels, continuous_predictions=None, continuous_labels=None,
                     average_difference=None, std_dev=None):
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
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Predictions: {predictions}\n")
            f.write(f"Real labels: {real_labels}\n")
            if continuous_predictions:
                f.write(f"Continuous Predictions: {continuous_predictions}\n")
            if continuous_labels:
                f.write(f"Continuous Labels: {continuous_labels}\n")
            if average_difference:
                f.write(f"Average Difference: {average_difference}\n")
            if std_dev:
                f.write(f"Standard Deviation: {std_dev}\n")
            f.write(f"Image shape: {str(self.image_shape)}\n")

    def test_model(self, request_from_server=False):
        if self.is_regression:
            self.test_filenames, self.test_labels, self.test_images = self.get_regression_values(
                self.test_filenames, self.test_labels, self.test_images)

        if request_from_server:
            self.model.load_model_weights()

        print("Evaluating model...")

        real_labels = []
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
            if self.is_regression:
                most_common_image_prediction = statistics.mean(patch_predictions)
            else:
                most_common_image_prediction = self.most_common_number(patch_predictions)
            predictions.append(most_common_image_prediction)
            real_labels.append(label)

        if self.is_regression:
            continuous_predictions = predictions.copy()
            predictions = map_to_age_group(np.array(predictions))
            print(f"Continuous Predictions: {[round(num) for num in continuous_predictions]}")
            continuous_labels = real_labels.copy()
            real_labels = map_to_age_group(real_labels)
            print(f"Continuous Labels:      {continuous_labels}")
            average_difference = mean_absolute_error(continuous_labels, continuous_predictions)
            print(f"Average Difference: {average_difference}")
            std_dev = np.std(continuous_predictions)
            print(f"Standard Deviation: {std_dev}")

        accuracy = accuracy_score(real_labels, predictions)
        print(f"Accuracy: {accuracy}")
        print(f"Predictions: {predictions}")
        print(f"Real labels: {real_labels}")
        if not request_from_server:
            self.confusion_matrix_generator.build_confusion_matrix_plot(real_labels,
                                                                        predictions,
                                                                        self.model_name,
                                                                        self.data_name)
        if self.is_regression:
            self.save_run_txt(accuracy, predictions, real_labels, continuous_predictions, continuous_labels,
                              average_difference, std_dev)
        else:
            self.save_run_txt(accuracy, predictions, real_labels)

        return predictions


def map_to_age_group(predictions, bins=(15, 25, 50)):
    res = []
    for prediction in predictions:
        # Initialize group index
        group = len(bins)  # Default to the last group if prediction exceeds all bins

        # Determine the correct bin for the prediction
        for i in range(len(bins)):
            if prediction <= bins[i]:
                group = i
                break

        res.append(group)
    return res


def construct_HHD_engine(base_dir, image_shape, request_from_server=False) -> Engine:
    # Setting file system
    train_path = base_dir / "train"
    test_path = base_dir / "test"
    csv_label_path = str(base_dir / "AgeSplit.csv")

    # Initializing engine
    engine = Engine(image_shape)
    if not request_from_server:
        # Setting engine labels & paths
        HHD_labels = LabelSplitter(csv_label_path)  # returns object with 'train', 'test', 'val' attributes
        engine.set_train_labels(HHD_labels.train)
        engine.set_test_labels(HHD_labels.test)
        engine.set_test_data_path(str(test_path))
        engine.set_train_data_path(str(train_path))

        engine.load_images(data_type='train', image_filename_col='File', label_col='Age')
        engine.load_images(data_type='test', image_filename_col='File', label_col='Age')

    return engine


def construct_KHATT_engine(base_dir, image_shape, request_from_server=False) -> Engine:
    # Setting file system
    train_path = base_dir / "Train"
    test_path = base_dir / "Test"
    csv_label_path = str(base_dir / "DatabaseStatistics-v1.0-NN.csv")

    # Initializing engine
    engine = Engine(image_shape)
    if not request_from_server:
        # Setting engine labels & paths
        KHATT_labels = LabelSplitter(csv_label_path, "Group", "R", "T", "V")  # gets 'train', 'test', 'val' attributes
        engine.set_train_labels(KHATT_labels.train)
        engine.set_test_labels(KHATT_labels.test)
        engine.set_test_data_path(str(test_path))
        engine.set_train_data_path(str(train_path))

        engine.load_images(data_type='train', image_filename_col='Form Number',
                           label_col='Age (1,2,3,or 4 from right to left)', clean_method=KHATT)
        engine.load_images(data_type='test', image_filename_col='Form Number',
                           label_col='Age (1,2,3,or 4 from right to left)', clean_method=KHATT)

    return engine
