from pathlib import Path

import numpy as np

from api_models import ModelResults
from src.data.path_variables import SRC, DATA, HHD
from src.engine import construct_hhd_engine
from transformer.create_model import create_vit_model


def transform_predict(image) -> ModelResults:
    num_classes = 4
    model = create_vit_model(num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    engine = construct_hhd_engine(
        base_dir=Path.cwd().parent / SRC / DATA / HHD,
        image_shape=(224, 224, 1),
        is_transformer=True,
        request_from_server=True
    )
    proc_images, proc_labels = engine.preprocess_data(images=[image], labels=[1], data_type='test')
    model.load_weights(
        "C:\\Users\\tomer\\PycharmProjects\\Final_Project\\src\\models\\model_weights\\transformer.h5")
    probability = model.predict(np.array(proc_images))
    predictions = np.argmax(probability, axis=1)
    return ModelResults(model_name="transformer",
                        data_set="any",
                        predictions=predictions)
