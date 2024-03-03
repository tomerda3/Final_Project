import pandas as pd


class DataSplitter:
    def __init__(self, directory_path: str, set_name: str = 'Set'):
        self.directory_path = directory_path
        df = pd.read_csv(self.directory_path)

        self._train_df = df[df[set_name] == 'train']
        self._test_df = df[df[set_name] == 'test']
        self._val_df = df[df[set_name] == 'val']

    @property
    def train(self):
        return self._train_df

    @property
    def test(self):
        return self._test_df

    @property
    def val(self):
        return self._val_df

