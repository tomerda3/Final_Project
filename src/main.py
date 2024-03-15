from engine import *
from pathlib import Path

VGG16 = "vgg16"
VGG19 = "vgg19"
XCEPTION = "xception"

if __name__ == "__main__":

    model = XCEPTION

    # Creating a new engine
    engine = get_bundled_engine(
        base_dir=Path.cwd() / "data",
        train_images_folder="train",
        test_images_folder="test",
        labels_file="AgeSplit.csv",
        image_shape=(400, 400, 3)
    )

    # Loading images to engine
    engine.load_images('train', 'File', 'Age')
    engine.load_images('test', 'File', 'Age')

    # Setting engine model
    engine.choose_model(model)

    # Training model
    engine.train_model()

    # Test model
    engine.test_model()
