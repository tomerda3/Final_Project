import numpy as np

from transformer.create_model import create_vit_model

input_shape = (224, 224, 3)
num_classes = 4
model = create_vit_model(num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
x_train = np.random.rand(10, *input_shape).astype(np.float32)  # here pass the images for train
y_train = np.random.randint(num_classes, size=(10,))  # here pass the labels for train

# Train model
model.fit(x_train, y_train, epochs=5)  # run the model

# test the model for predictions
y_test = x_train = np.random.rand(3, *input_shape).astype(np.float32)
y_train = np.random.randint(num_classes, size=(3,))

predictions = model.predict(y_test)
print(predictions)