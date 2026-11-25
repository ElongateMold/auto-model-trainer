import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.base import BaseModelTrainer


class TreeModelTrainer(BaseModelTrainer):
    def build_models(self) -> dict[str, BaseEstimator]:
        if self.problem_type == "classification":
            return {
                "CART_Classifier": DecisionTreeClassifier(random_state=self.random_state),
                "RandomForest_Classifier": RandomForestClassifier(random_state=self.random_state),
            }
        return {
            "CART_Regressor": DecisionTreeRegressor(random_state=self.random_state),
            "RandomForest_Regressor": RandomForestRegressor(random_state=self.random_state),
        }

    def get_param_grids(self) -> dict[str, dict]:
        if self.problem_type == "classification":
            return {
                "CART_Classifier": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [3, 5, 8, None],
                    "min_samples_split": [2, 10],
                    "ccp_alpha": [0.0, 0.005, 0.01],
                },
                "RandomForest_Classifier": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                },
            }
        return {
            "CART_Regressor": {
                "criterion": ["squared_error", "absolute_error"],
                "max_depth": [3, 5, 8, None],
                "min_samples_split": [2, 10],
                "ccp_alpha": [0.0, 0.005, 0.01],
            },
            "RandomForest_Regressor": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
            },
        }

    def get_feature_importances(
        self, feature_names: list[str], model_name: str | None = None
    ) -> pd.DataFrame:
        if not self.best_models:
            raise RuntimeError("Fit the tree models before extracting feature importances.")

        chosen_name = model_name if model_name is not None else next(iter(self.best_models))
        if chosen_name not in self.best_models:
            raise ValueError(f"Model '{chosen_name}' not found in trained models.")

        estimator = self.best_models[chosen_name]
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            raise AttributeError(f"'{chosen_name}' does not expose feature_importances_.")

        return (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values(by="importance", ascending=False)
            .reset_index(drop=True)
        )