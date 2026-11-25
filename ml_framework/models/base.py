from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


class BaseModelTrainer(ABC):
    def __init__(self, problem_type: str, cv_splits: int = 5, random_state: int = 42):
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem_type: '{problem_type}'")
        self.problem_type = problem_type
        self.cv_splits = cv_splits
        self.random_state = random_state

        self.models: dict[str, BaseEstimator] = self.build_models()
        self.param_grids: dict[str, dict] = self.get_param_grids()
        self.best_models: dict[str, BaseEstimator] = {}
        self.cv_results_summary: dict[str, dict] = {}

    @abstractmethod
    def build_models(self) -> dict[str, BaseEstimator]:
        """Returns {model_name: unfitted_estimator}."""
        pass

    @abstractmethod
    def get_param_grids(self) -> dict[str, dict]:
        """Returns {model_name: hyperparameter_grid} for GridSearchCV."""
        pass

    def _get_cv_splitter(self):
        if self.problem_type == "classification":
            return StratifiedKFold(
                n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
            )
        return KFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        cv_splitter = self._get_cv_splitter()
        scoring = "f1_weighted" if self.problem_type == "classification" else "r2"

        for name, estimator in self.models.items():
            grid = self.param_grids.get(name, {})

            search = GridSearchCV(
                estimator=estimator,
                param_grid=grid,
                cv=cv_splitter,
                scoring=scoring,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)

            self.best_models[name] = search.best_estimator_
            self.cv_results_summary[name] = {
                "best_score": round(float(search.best_score_), 4),
                "best_params": search.best_params_,
            }

    def predict(self, X_test: pd.DataFrame) -> dict[str, pd.Series]:
        if not self.best_models:
            raise RuntimeError("Call train() before calling predict().")

        predictions: dict[str, pd.Series] = {}
        for name, best_model in self.best_models.items():
            preds = best_model.predict(X_test)
            predictions[name] = pd.Series(preds, index=X_test.index)
        return predictions


def get_model_trainer(model_family: str, problem_type: str) -> BaseModelTrainer:
    from models.supervised.linear_models import LinearModelTrainer
    from models.supervised.prototype_models import PrototypeModelTrainer
    from models.supervised.spline_models import SplineModelTrainer
    from models.supervised.tree_models import TreeModelTrainer

    trainers: dict[str, type[BaseModelTrainer]] = {
        "linear": LinearModelTrainer,
        "prototype": PrototypeModelTrainer,
        "spline": SplineModelTrainer,
        "tree_based": TreeModelTrainer,
    }

    trainer_cls = trainers.get(model_family)
    if trainer_cls is None:
        raise ValueError(f"Unsupported model family: '{model_family}'")
    return trainer_cls(problem_type)