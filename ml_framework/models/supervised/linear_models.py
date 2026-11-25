from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from models.base import BaseModelTrainer


class LinearModelTrainer(BaseModelTrainer):
    def build_models(self) -> dict[str, BaseEstimator]:
        if self.problem_type == "classification":
            return {
                "LogisticRegression": LogisticRegression(
                    solver="saga", max_iter=2000, random_state=self.random_state
                ),
                "RidgeClassifier": RidgeClassifier(random_state=self.random_state),
            }
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=self.random_state),
            "Lasso": Lasso(max_iter=2000, random_state=self.random_state),
        }

    def get_param_grids(self) -> dict[str, dict]:
        if self.problem_type == "classification":
            return {
                "LogisticRegression": {
                    "penalty": ["l1", "l2"],
                    "C": [0.1, 1.0, 10.0],
                },
                "RidgeClassifier": {
                    "alpha": [0.1, 1.0, 10.0],
                },
            }
        return {
            "LinearRegression": {},
            "Ridge": {
                "alpha": [0.1, 1.0, 10.0],
            },
            "Lasso": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
            },
        }