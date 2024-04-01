from engine import *
from pathlib import Path
from models import model_names
import tensorflow as tf

DATA = 'data'
HHD = 'HHD'
KHATT = 'KHATT'

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5600)])

    # Construct HHD engine
    HHD_engine = construct_HHD_engine(
        base_dir=Path.cwd() / "data" / "HHD",
        image_shape=(400, 400, 1)
    )

    # Setting engine model
    HHD_engine.set_model(model_names.ResNet)

    # Training model
    HHD_engine.train_model()

    # Test model
    HHD_engine.test_model()

    # # Construct KHATT engine
    # KHATT_engine = construct_KHATT_engine(
    #     base_dir=Path.cwd() / "data" / "KHATT",
    #     image_shape=(500, 500, 1)
    # )
    #
    # # Setting engine model
    # KHATT_engine.set_model(model_names.ResNet)
    #
    # # Training model
    # KHATT_engine.train_model()
    #
    # # Test model
    # KHATT_engine.test_model()
