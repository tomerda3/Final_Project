from engine import Engine
from pathlib import Path
from models.model_names import *

if __name__ == "__main__":

    # Construct HHD engine
    HHD_engine = Engine.construct_HHD_engine(
        base_dir=Path.cwd() / "data" / "HHD",
        image_shape=(400, 400, 1)
    )

    # Setting engine model
    HHD_engine.choose_model(ResNet)

    # Training model
    HHD_engine.train_model()

    # Test model
    HHD_engine.test_model()

    # # Construct KHATT engine
    # KHATT_engine = Engine.construct_KHATT_engine(
    #     base_dir=Path.cwd() / "data" / "KHATT",
    #     image_shape=(400, 400, 1)
    # )
    #
    # # Setting engine model
    # KHATT_engine.choose_model(Xception)
    #
    # # Training model
    # KHATT_engine.train_model()
    #
    # # Test model
    # KHATT_engine.test_model()
