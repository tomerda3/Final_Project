import engine
from pathlib import Path

VGG16 = "vgg16"
VGG19 = "vgg19"
XCEPTION = "xception"

if __name__ == "__main__":

    model = XCEPTION

    # Creating a new engine
    e = engine.get_bundled_engine(
        base_dir=Path.cwd(),
        data_dir="data",
        train_dir="train",
        test_dir="test",
        labels_file="AgeSplit.csv",
        image_shape=(400, 400, 3)
    )

    # Loading images to engine
    e.load_images('train', 'File', 'Age')
    e.load_images('test', 'File', 'Age')

    # Setting engine model
    e.choose_model(model)

    # Training model
    e.train_model()

    # Test model
    e.test_model()
