from engine import *
from pathlib import Path
from models import models_metadata
from data.path_variables import *
from data_handler.datasets_constructors import constructors


def run_engine(dataset, image_size, chosen_model):
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

def run_all_configs():
    construct_engine = {
        HHD: construct_hhd_engine,
        KHATT: construct_khatt_engine
    }

    for dataset in [HHD, KHATT]:
        print(f"Run started for dataset: {dataset}")
        main_engine = construct_engine[dataset](
            base_dir=Path.cwd() / DATA / dataset,
            image_shape=(500, 500, 1)
        )

        i = 0
        while i < len(models_metadata.models_list):
            model = models_metadata.models_list[i]
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