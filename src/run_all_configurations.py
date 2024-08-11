from engine import *
from pathlib import Path
from models import model_names
from data.path_variables import *


def run_all_configs():
    construct_engine = {
        HHD: construct_HHD_engine,
        KHATT: construct_KHATT_engine
    }

    for dataset in [HHD, KHATT]:
        print(f"Run started for dataset: {dataset}")
        main_engine = construct_engine[dataset](
            base_dir=Path.cwd() / DATA / dataset,
            # image_shape=(200, 200, 1)
            image_shape=(500, 500, 1)
        )

        # for model in model_names.models_list:
        i = 0
        while i < len(model_names.models_list):
            model = model_names.models_list[i]
            try:
                print(f"Running {model} model on {dataset} dataset.")
                # Setting engine model
                main_engine.set_model(model)

                # Training model
                main_engine.train_model()

                # Test model
                main_engine.test_model()

                i += 1

            except:
                print("Run failed! Trying again...")

def run_HHD_convnextxl():

    sizes = [1000, 600]

    for num in reversed(sorted(sizes)):

        try:
            model_name = model_names.ConvNeXtXLarge

            print(f"Running {model_name} model on HHD dataset with size {num}x{num}.")

            engine = construct_HHD_engine(
                base_dir=Path.cwd() / DATA / HHD,
                image_shape=(num, num, 1)
            )

            # Setting engine model
            engine.set_model(model_name)

            # Training model
            engine.train_model()

            # Test model
            engine.test_model()

        except:
            print("Run failed, jumping to the next one!")
            continue

