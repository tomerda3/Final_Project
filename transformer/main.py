import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from src.data.path_variables import DATA, HHD, SRC
from src.engine import construct_hhd_engine
from transformer.create_model import create_vit_model

input_shape = (224, 224, 1)
num_classes = 3
model = create_vit_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# construct engine for hhd
engine = construct_hhd_engine(
    base_dir=Path.cwd().parent / SRC / DATA / HHD,
    image_shape=(224, 224, 1),
    is_transformer=True
)

x_train = engine.train_images
y_train = engine.train_labels

x_test = engine.test_images
y_test = engine.test_labels

model.fit(np.array(x_train), np.array(y_train), epochs=10, batch_size=32)

probability = model.predict(np.array(x_test))
predictions = np.argmax(probability, axis=1)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(f"Predictions: {predictions}")
model.save_weights(f"/Users/ofri/Documents/GitHub/Final_Project/src/models/model_weights/transformer.weights.h5")
