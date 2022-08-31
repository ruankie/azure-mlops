from __future__ import annotations

import logging
import random
import pandas as pd


class DummyClassifier:
    """Dummy model class with access to fit and predict methods that does nothing.
    Prediction will return a randomly selected integer (0 or 1) representing a binary class.
    """

    def __init__(self) -> None:
        logging.info("Initialising model...")
        self.params = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
        """Mimics training model on training data set (doesn't actually use data)."""
        logging.info("Mimicking fitting model...")
        self.params = {"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "delta": 0.4}
        return self

    def predict(self, X_test: pd.DataFrame) -> DummyClassifier:
        """Mimics predicting binary class for test data (doesn't actually use data)."""
        logging.info("Mimicking creating predictions from model...")
        return random.choice([0, 1])
