import numpy as np
import pandas as pd
from pathlib import Path


class Dataset:
    def __init__(self, path: str, label: str):
        self.df = None
        self.x = None
        self.y = None

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"file {path} not found.")

        self._read_dataset(path)
        self._extract_features_label(label)

    def _read_dataset(self, path: Path):
        match path.suffix:
            case ".xlsx":
                self.df = pd.read_excel(path)
            case ".csv":
                self.df = pd.read_csv(path)
            case _:
                raise NotImplementedError(f"dataset format '{path.suffix}' is not supported yet.")

    def _extract_features_label(self, label: str):
        self.x = self.df.drop(columns=[label])
        self.y = self.df[label]

    def drop(self, column: str):
        self.x = self.x.drop(columns=[column])
        return self

    @staticmethod
    def serialize(features: pd.DataFrame):
        return Dataset._serialize_v1(features)

    @staticmethod
    def _serialize_v1(features: pd.DataFrame):
        return [
            ", ".join([
                f'{column}:{value:.2f}' if type(value) is float else f'{column}:{value}'
                for column, value in zip(features.columns, row[1])
            ])
            for row in features.iterrows()
        ]

    def train_test_split(self, split: float | int):
        if 0 < split < 1:
            split *= 100

        if 0 < split < 100:
            b = int(split * len(self.x) / 100)
            train_x, train_y = self.x[:b], self.y[:b]
            test_x, test_y = self.x[b:], self.y[b:]

            return train_x, test_x, train_y, test_y
        else:
            raise ValueError("split should be either 0 < split < 1 or 0 < split < 100")
