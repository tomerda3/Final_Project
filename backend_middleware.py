from pathlib import Path

from api_models import ModelResults
from src.data.path_variables import DATA, SRC
from src.models.model_names import models_list
from src.engine import construct_HHD_engine, construct_KHATT_engine


class BackendMiddleware:

    def __init__(self):
        self.models = models_list
        self.config = {"HHD": construct_HHD_engine, "KHATT": construct_KHATT_engine}

    def create_and_run_engine(self, data_set: str, model_name: str, image, image_shape=(500, 500, 1)):
        engine = self.config[data_set](image_shape=image_shape,
                                       base_dir=Path.cwd() / SRC / DATA / data_set)
        engine.set_model(model_name)
        proc_images, proc_labels = engine.preprocess_data(images=[image], labels=[1], data_type='test') # ignore the
        # label just to not break the api
        engine.test_images = proc_images
        engine.test_labels = proc_labels
        predictions = engine.test_model(request_from_server=True)
        return self._build_results(predictions, model_name, data_set)

    def _build_results(self, predictions: str, model_name: str, data_set: str) -> ModelResults:
        return ModelResults(model_name=model_name,
                            data_set=data_set,
                            predictions=predictions)
