from typing import List
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.utils import to_categorical

NUM_OF_CLASSES = 4


class Model:

    def train_model(self, train_data, train_labels: List):
        cv2.imwrite("test.png", train_data[0])
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(learning_rate=0.001),
                           metrics=['accuracy'])

        fittable_labels = to_categorical(train_labels, num_classes=NUM_OF_CLASSES)

        print("Epochs excluding base layers...")
        self.model.fit(x=train_data, y=fittable_labels,
                       epochs=30,
                       batch_size=32)

        for layer in self.model.layers:
            layer.trainable = True

        print("Epochs including base layers...")
        self.model.fit(x=train_data, y=fittable_labels,
                       epochs=30,
                       batch_size=32)

    def patch_evaluation(self, patches):
        # try:
        probabilities = self.model.predict(patches)
        predictions = np.argmax(probabilities, axis=1)
        return predictions
        # except BaseException as e:
        #     print(e)
        #     return [4, 4, 4, 4]
