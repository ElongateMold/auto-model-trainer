from abc import ABC, abstractmethod
from pathlib import Path
import warnings
import os
import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, r2_score
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
        """Returns {model_name: hyperparameter_grid} for cross-validation."""
        pass

    def _get_cv_splitter(self):
        if self.problem_type == "classification":
            return StratifiedKFold(
                n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
            )
        return KFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseModelTrainer":
        cv_splitter = self._get_cv_splitter()
        scoring = "f1_weighted" if self.problem_type == "classification" else "r2"

        os.environ["PYTHONWARNINGS"] = "ignore"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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

        return self

    def predict(self, X_test: pd.DataFrame) -> dict[str, pd.Series]:
        if not self.best_models:
            raise RuntimeError("Call fit() before calling predict().")

        predictions: dict[str, pd.Series] = {}
        for name, best_model in self.best_models.items():
            preds = best_model.predict(X_test)
            predictions[name] = pd.Series(preds, index=X_test.index)
        return predictions

    def score(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        predictions = self.predict(X_test)
        scores: dict[str, float] = {}

        for name, y_pred in predictions.items():
            if self.problem_type == "classification":
                val = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            else:
                val = r2_score(y_test, y_pred)
            scores[name] = round(float(val), 4)

        return scores

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        if not self.best_models:
            raise RuntimeError("Call fit() before saving trained models.")

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[str, Path] = {}

        for name, best_model in self.best_models.items():
            file_path = target_dir / f"{name}.joblib"
            joblib.dump(best_model, file_path)
            saved_paths[name] = file_path

        return saved_paths


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