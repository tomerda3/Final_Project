from pathlib import Path
from models import model_names
from data.path_variables import *
import tensorflow as tf
from data_handler.datasets_constructors import constructors

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
    chosen_model = model_names.ConvNeXtXLargeRegression

    # Construct engine
    main_engine = constructors[dataset](
        base_dir=Path.cwd() / DATA / dataset,
        image_shape=(image_size, image_size, 1)
    )

    # Setting engine model
    main_engine.set_model(chosen_model)

    # Training model
    main_engine.train_model()

    # Test model
    main_engine.test_model()
