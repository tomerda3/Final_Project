from models import models_metadata
from data.path_variables import *
import tensorflow as tf
from engine_runner import run_engine, run_all_configs

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

    # Choose dataset
    dataset = HHD  # Options are: HHD / KHATT

    # Choose image size
    image_size = 600  # Typically 200~1000

    # Choose model:
    chosen_model = models_metadata.ConvNeXtXLarge

    run_engine(dataset, image_size, chosen_model)

    # run_all_configs()