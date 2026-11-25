from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from models.base import BaseModelTrainer


class TreeModelTrainer(BaseModelTrainer):
    def build_models(self) -> dict[str, BaseEstimator]:
        if self.problem_type == "classification":
            return {
                "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            }
        return {
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        }