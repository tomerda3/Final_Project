from engine import *
from pathlib import Path
from models.model_names import *

if __name__ == "__main__":

    # Creating a new engine
    engine = get_bundled_engine(
        base_dir=Path.cwd() / "data",
        train_images_folder="train",
        test_images_folder="test",
        labels_file="AgeSplit.csv",
        image_shape=(400, 400, 3)
    )

    # Loading images to engine
    engine.load_images(data_type='train', image_filename_col='File', label_col='Age')
    engine.load_images(data_type='test', image_filename_col='File', label_col='Age')

    # Setting engine model
    engine.choose_model(XCEPTION)

    # Training model
    engine.train_model()

    # Test model
    engine.test_model()
