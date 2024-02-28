from typing import Tuple, List
from keras.applications import VGG19
from keras.optimizers import Adam

class Vgg19:
    def run_model(self, train_data: List, train_labels: List, validation_data: List, validation_labels: List):
        # Compile the model with appropriate loss, optimizer, and metrics
        self.model.compile(loss='categorical_crossentropy',  # Assuming multi-class classification
                            optimizer=Adam(lr=0.001),  # Set a learning rate
                            metrics=['accuracy'])

        # Train the model on the training data
        history = self.model.fit(train_data, train_labels,
                                  epochs=10,  # Set the number of epochs
                                  batch_size=32,  # Adjust batch size as needed
                                  validation_data=(validation_data, validation_labels))

        # Evaluate the model's performance on validation data
        loss, accuracy = self.model.evaluate(validation_data, validation_labels)
        print('Validation Loss:', loss)
        print('Validation Accuracy:', accuracy)

    def __init__(self, weights: str, include_top: bool, input_shape: Tuple):
        self.model = VGG19(weights=weights, include_top=include_top, input_shape=input_shape)
        self.fine_tune(False)

