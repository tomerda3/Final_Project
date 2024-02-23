from typing import Tuple, List

from keras.applications import VGG16


class Vgg16:

    def __init__(self, weights: str, include_top: bool, input_shape: Tuple):
        self.model = VGG16(weights=weights, include_top=include_top, input_shape=input_shape)

    def fine_tune(self, freeze_pre_trained_layers: bool):
        if freeze_pre_trained_layers:
            for layer in self.model.layers:
                layer.trainable = False

    def run_model(self, train_data: List, train_labels: List, validation_data: List, validation_labels: List):
        pass
