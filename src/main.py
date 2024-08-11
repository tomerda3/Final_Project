from engine import *
from pathlib import Path
from models import model_names
from data.path_variables import *
import tensorflow as tf
from run_all_configurations import run_all_configs, run_HHD_convnextxl

if __name__ == "__main__":
    # Setting up GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6400)])
        print("GPU detected.")
    else:
        print("No GPU detected.")

    # run_all_configs()

    run_HHD_convnextxl()

    # # Construct engine
    # main_engine = construct_HHD_engine(
    #     base_dir=Path.cwd() / DATA / HHD,
    #     image_shape=(1000, 1000, 1)
    # )
    # # main_engine = construct_KHATT_engine(
    # #     base_dir=Path.cwd() / DATA / KHATT,
    # #     image_shape=(500, 500, 1)
    # # )

    # # Setting engine model
    # main_engine.set_model(model_names.ConvNeXtXLarge)
    #
    # # Training model
    # main_engine.train_model()
    #
    # # Test model
    # main_engine.test_model()
