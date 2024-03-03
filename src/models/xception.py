from typing import Tuple, List
from keras.optimizers import Adam
from keras.applications import Xception
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras import layers
import tensorflow as tf


class XceptionModel:

    def __init__(self, weights: str = "imagenet", include_top: bool = True, input_shape: Tuple = (0, 0)):
        # self.model = Xception(weights=weights, include_top=include_top, input_shape=input_shape)

        # Define your custom input layer
        inputs = layers.Input(shape=(400, 400, 3))

        # Add a resizing layer to match Xception's input shape
        x = layers.Resizing(299, 299)(inputs)

        # Rest of the Xception model
        x = Xception(weights=weights, include_top=include_top)(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)

    def run_model(self, train_data: List, train_labels: List):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=0.001),
                           metrics=['accuracy'])

        # print(self.model.summary())

        self.model.fit(x=train_data, y=train_labels,
                       epochs=10,
                       batch_size=32)

    def evaluation(self, test_data, test_labels):
        predictions = self.model.predict(test_data)

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
