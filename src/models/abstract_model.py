from typing import List
import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical

NUM_OF_CLASSES = 4


class Model:

    def train_model(self, train_data, train_labels: List):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=0.001),
                           metrics=['accuracy'])

        fittable_labels = to_categorical(train_labels, num_classes=NUM_OF_CLASSES)

        print("Epochs excluding base layers...")
        self.model.fit(x=train_data, y=fittable_labels,
                       epochs=10,
                       batch_size=32)

        for layer in self.model.layers:
            layer.trainable = True

        print("Epochs including base layers...")
        self.model.fit(x=train_data, y=fittable_labels,
                       epochs=20,
                       batch_size=32)

    def patch_evaluation(self, patches):
        probabilities = self.model.predict(patches)
        predictions = np.argmax(probabilities, axis=1)

        return predictions
