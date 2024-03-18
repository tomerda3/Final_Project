from typing import Tuple, List

import numpy as np
from keras.optimizers import Adam
from keras.applications import Xception
from keras.src.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras import layers
from keras.utils import to_categorical
import keras
import tensorflow as tf

NUM_OF_CLASSES = 4


class XceptionModel:

    def __init__(self, weights: str = "imagenet", include_top: bool = True, input_shape: Tuple = (0, 0)):
        # inputs = layers.Input(shape=(input_shape[0], input_shape[1], 3))
        inputs = layers.Input(shape=input_shape)

        # optimal_layer = layers.Resizing(299, 299)(inputs)

        if include_top:  # If using pre-trained top layers that expect RGB
            gray_to_rgb = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)
            optimal_layer = layers.Resizing(299, 299)(gray_to_rgb)
        else:  # If not using pre-trained top layers, keep as grayscale
            optimal_layer = layers.Resizing(299, 299)(inputs)

        base_model = Xception(weights=weights, include_top=include_top)
        for layer in base_model.layers:
            layer.trainable = False
        base_model = base_model(optimal_layer)
        bridge_layer_to_output = Dense(512, activation='relu')(base_model)
        output_layer = Dense(units=NUM_OF_CLASSES, activation='softmax')(bridge_layer_to_output)
        self.model = keras.Model(inputs=inputs, outputs=output_layer)

    def train_model(self, train_data, train_labels: List):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=0.001),
                           metrics=['accuracy'])

        fittable_labels = to_categorical(train_labels, num_classes=NUM_OF_CLASSES)

        print("Epochs excluding base layers...")
        self.model.fit(x=train_data, y=fittable_labels,
                       epochs=20,
                       batch_size=32)

        for layer in self.model.layers:
            layer.trainable = True

        print("Epochs including base layers...")
        self.model.fit(x=train_data, y=fittable_labels,
                       epochs=20,
                       batch_size=32)

    # def evaluation(self, test_data, test_labels):  # TODO: Remove this after the whole team agrees
    #     probabilities = self.model.predict(test_data)
    #     predictions = np.argmax(probabilities, axis=1)
    #
    #     accuracy = accuracy_score(test_labels, predictions)
    #     # precision = precision_score(test_labels, predictions, average=None)
    #     # recall = recall_score(test_labels, predictions, average=None)
    #
    #     print("Accuracy:", accuracy)
    #     # print("Precision:", precision)
    #     # print("Recall:", recall)
    #
    #     print("Predictions: ", list(predictions))
    #     print("Real labels: ", list(test_labels))

    def patch_evaluation(self, patches):
        probabilities = self.model.predict(patches)
        predictions = np.argmax(probabilities, axis=1)

        return predictions
