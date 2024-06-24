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
            image_shape=(200, 200, 1)
        )

        for model in model_names.models_list:
            print(f"Running {model} model on {dataset} dataset.")
            # Setting engine model
            main_engine.set_model(model)

            # Training model
            main_engine.train_model()

            # Test model
            main_engine.test_model()
