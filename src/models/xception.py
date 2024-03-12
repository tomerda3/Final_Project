from typing import Tuple, List

import numpy as np
from keras.optimizers import Adam
from keras.applications import Xception
from keras.src.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras import layers
from keras.utils import to_categorical
import keras

NUM_OF_CLASSES = 4

class XceptionModel:

    # TODO: Freeze base model
    def __init__(self, weights: str = "imagenet", include_top: bool = True, input_shape: Tuple = (0, 0)):
        inputs = layers.Input(shape=(input_shape[0], input_shape[1], 3))
        optimal_layer = layers.Resizing(299, 299)(inputs)
        base_model = Xception(weights=weights, include_top=include_top)
        for layer in base_model.layers:
            layer.trainable = False
        base_model = base_model(optimal_layer)
        bridge_layer_to_output = Dense(512, activation='relu')(base_model)
        output_layer = Dense(units=NUM_OF_CLASSES, activation='softmax')(bridge_layer_to_output)
        self.model = keras.Model(inputs=inputs, outputs=output_layer)


    def run_model(self, train_data, train_labels: List):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=0.001),
                           metrics=['accuracy'])

        fittable_labels = to_categorical(train_labels, num_classes=NUM_OF_CLASSES)

        data_cut = len(train_data) // 5  # 20% to the freezed model, then 80% to the unfreezed model

        freezed_model_train_data, freezed_model_fittable_labels = train_data[:data_cut], fittable_labels[:data_cut]
        unfreezed_model_train_data, unfreezed_model_fittable_labels = train_data[data_cut:], fittable_labels[data_cut:]


        self.model.fit(x=freezed_model_train_data, y=freezed_model_fittable_labels,
                       epochs=10,
                       batch_size=32)

        # TODO: Unfreeze base model
        for layer in self.model.layers:
            layer.trainable = True

        self.model.fit(x=unfreezed_model_train_data, y=unfreezed_model_fittable_labels,
                       epochs=10,
                       batch_size=32)


    def evaluation(self, test_data, test_labels):
        probabilities = self.model.predict(test_data)
        predictions = np.argmax(probabilities, axis=1)

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average=None)
        recall = recall_score(test_labels, predictions, average=None)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
