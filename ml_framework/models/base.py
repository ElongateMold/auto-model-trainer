from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator


class BaseModelTrainer(ABC):
    def __init__(self, problem_type: str):
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem_type: '{problem_type}'")
        self.problem_type = problem_type
        self.models: dict[str, BaseEstimator] = self.build_models()

    @abstractmethod
    def build_models(self) -> dict[str, BaseEstimator]:
        """Each subclass returns a dictionary of {model_name: sklearn_model}."""
        pass

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fits every model in self.models on the preprocessed training data."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame) -> dict[str, pd.Series]:
        """Generates predictions for every trained model on X_test."""
        predictions: dict[str, pd.Series] = {}
        for name, model in self.models.items():
            preds = model.predict(X_test)
            predictions[name] = pd.Series(preds, index=X_test.index)
        return predictions

def get_model_trainer(model_family: str, problem_type: str) -> BaseModelTrainer:
    from models.supervised.linear_models import LinearModelTrainer
    from models.supervised.tree_models import TreeModelTrainer
    # Later you can import PrototypeModelTrainer and SplineModelTrainer here too

    if model_family == "linear":
        return LinearModelTrainer(problem_type)
    elif model_family == "tree_based":
        return TreeModelTrainer(problem_type)
    else:
        raise ValueError(f"Unsupported model family: '{model_family}'")