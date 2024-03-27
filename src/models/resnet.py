from typing import Tuple
from .abstract_model import Model
from keras.applications import ResNet50
from keras.src.layers import Dense
from keras import layers
import keras
import tensorflow as tf

NUM_OF_CLASSES = 4


class ResNet50Model(Model):

    def __init__(self, weights: str = "imagenet", include_top: bool = True, input_shape: Tuple = (0, 0)):
        inputs = layers.Input(shape=input_shape)

        if include_top:  # If using pre-trained top layers that expect RGB
            gray_to_rgb = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)
            optimal_layer = layers.Resizing(224, 224)(gray_to_rgb)  # Adjusted size for ResNet50
        else:  # If not using pre-trained top layers, keep as grayscale
            optimal_layer = layers.Resizing(224, 224)(inputs)  # Adjusted size for ResNet50

        base_model = ResNet50(weights=weights, include_top=include_top)
        for layer in base_model.layers:
            layer.trainable = False
        base_model = base_model(optimal_layer)
        bridge_layer_to_output = Dense(512, activation='relu')(base_model)
        output_layer = Dense(units=NUM_OF_CLASSES, activation='softmax')(bridge_layer_to_output)
        self.model = keras.Model(inputs=inputs, outputs=output_layer)
