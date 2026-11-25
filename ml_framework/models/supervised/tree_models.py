import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.base import BaseModelTrainer


class TreeModelTrainer(BaseModelTrainer):
    def build_models(self) -> dict[str, BaseEstimator]:
        if self.problem_type == "classification":
            return {
                "CART_Classifier": DecisionTreeClassifier(random_state=self.random_state),
            }
        return {
            "CART_Regressor": DecisionTreeRegressor(random_state=self.random_state),
        }

    def get_param_grids(self) -> dict[str, dict]:
        if self.problem_type == "classification":
            return {
                "CART_Classifier": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [3, 5, 8, None],
                    "min_samples_split": [2, 10, 20],
                    "ccp_alpha": [0.0, 0.005, 0.01],  # Cost-Complexity Pruning
                }
            }
        return {
            "CART_Regressor": {
                "criterion": ["squared_error", "absolute_error"],
                "max_depth": [3, 5, 8, None],
                "min_samples_split": [2, 10, 20],
                "ccp_alpha": [0.0, 0.005, 0.01],
            }
        }

    def get_feature_importances(self, feature_names: list[str]) -> pd.DataFrame:
        if not self.best_models:
            raise RuntimeError("Train the CART model before extracting feature importances.")

        model_name = next(iter(self.best_models))
        best_tree = self.best_models[model_name]

        importances = getattr(best_tree, "feature_importances_", None)
        if importances is None:
            raise AttributeError(f"{model_name} does not expose feature_importances_.")

        report = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return report