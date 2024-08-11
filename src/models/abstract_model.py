from typing import List
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical
from src.data_handler.class_weights import class_weights

NUM_OF_CLASSES = 4


class Model:

    def train_model(self, train_data, train_labels: List, database_name: str = "NULL"):
        self.model.compile(loss='categorical_crossentropy',
                           # optimizer=Adam(learning_rate=0.001),
                           optimizer=Adam(learning_rate=0.0005),
                           metrics=['accuracy'])

        fittable_labels = to_categorical(train_labels, num_classes=NUM_OF_CLASSES)

        weights = None
        if database_name != "NULL":
            weights = class_weights[database_name]
        else:
            print("Invalid database name was sent to train")

        print("Epochs excluding base layers...")
        self.model.fit(x=train_data, y=fittable_labels,
                       # epochs=15,
                       epochs=20,
                       batch_size=32,
                       class_weight=weights)

        for layer in self.model.layers:
            layer.trainable = True

        print("Epochs including base layers...")
        self.model.fit(x=train_data, y=fittable_labels,
                       # epochs=30,
                       epochs=40,
                       batch_size=32,
                       class_weight=weights)
        self.save_model_weights()

    def patch_evaluation(self, patches):
        probabilities = self.model.predict(patches)  # TODO: fix sometimes patches is an empty sequence
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def save_model_weights(self):
        self.model.save_weights(f"./model_weights/{self.__class__.__name__}.h5")

    def load_model_weights(self):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=0.0005),
                           metrics=['accuracy'])
        self.model.load_weights(f"/Users/ofri/Documents/GitHub/Final_Project/src/models/model_weights/ConvNeXtXLargeModel.h5")
