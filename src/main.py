from engine import *
from pathlib import Path
from models import model_names
from data.path_variables import *
import tensorflow as tf

if __name__ == "__main__":

    # Setting up GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5600)])

    # Construct engine
    # main_engine = construct_HHD_engine(
    #     base_dir=Path.cwd() / DATA / HHD,
    #     image_shape=(400, 400, 1)
    # )
    main_engine = construct_KHATT_engine(
        base_dir=Path.cwd() / DATA / KHATT,
        image_shape=(500, 500, 1)
    )

    # Setting engine model
    main_engine.set_model(model_names.ResNet50)

    # Training model
    main_engine.train_model()

    # Test model
    main_engine.test_model()
