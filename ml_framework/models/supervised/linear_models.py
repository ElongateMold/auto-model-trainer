from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from models.base import BaseModelTrainer


class LinearModelTrainer(BaseModelTrainer):
    def build_models(self) -> dict[str, BaseEstimator]:
        if self.problem_type == "classification":
            return {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                "RidgeClassifier": RidgeClassifier(random_state=42),
            }
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0, random_state=42),
        }