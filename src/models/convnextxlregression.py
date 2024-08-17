from typing import Tuple, List
from .abstract_model import Model
from keras.applications import ConvNeXtXLarge
from keras.layers import Dense, GlobalAveragePooling2D
from keras import layers
import keras
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam


class ConvNeXtXLargeRegressionModel(Model):

    def __init__(self, weights: str = "imagenet", include_top: bool = True, input_shape: Tuple = (224, 224, 3)):
        inputs = layers.Input(shape=input_shape)

        if include_top:  # If using pre-trained top layers that expect RGB
            gray_to_rgb = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)
            optimal_layer = layers.Resizing(224, 224)(gray_to_rgb)  # Adjusted size for ConvNeXtXLarge
        else:  # If not using pre-trained top layers, keep as grayscale
            optimal_layer = layers.Resizing(224, 224)(inputs)  # Adjusted size for ConvNeXtXLarge

        self.x_large = ConvNeXtXLarge(weights=weights, include_top=False, input_shape=(224, 224, 3))
        base_model = self.x_large  # Use ConvNeXtXLarge model
        for layer in base_model.layers:
            layer.trainable = False
        base_model_output = base_model(optimal_layer)
        pooling_layer = GlobalAveragePooling2D()(base_model_output)
        bridge_layer_to_output = Dense(512, activation='relu')(pooling_layer)
        output_layer = Dense(units=1, activation='linear')(bridge_layer_to_output)  # Regression output
        self.model = keras.Model(inputs=inputs, outputs=output_layer)

    def train_model(self, train_data, train_labels: List, database_name: str = "NULL"):
        # Convert train_labels to a numpy array
        train_labels = np.array(train_labels)

        self.model.compile(loss='mean_squared_error',
                           optimizer=Adam(learning_rate=0.0005),
                           metrics=['mae'])

        print("Epochs excluding base layers...")
        self.model.fit(x=train_data, y=train_labels,
                       epochs=20,
                       batch_size=32)

        for layer in self.model.layers:
            layer.trainable = True

        print("Epochs including base layers...")
        self.model.fit(x=train_data, y=train_labels,
                       epochs=80,
                       batch_size=32)

    def patch_evaluation(self, patches):
        continuous_predictions = self.model.predict(patches)  # Predict continuous age values
        return [elem[0] for elem in continuous_predictions]
