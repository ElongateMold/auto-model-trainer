import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class ModelEvaluator:
    def __init__(self, problem_type: str):
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem_type: '{problem_type}'")
        self.problem_type = problem_type

    def _score_classification(self, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_weighted": round(
                float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4
            ),
            "recall_weighted": round(
                float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4
            ),
            "f1_weighted": round(
                float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
            ),
        }

    def _score_regression(self, y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }

    def evaluate_all(
        self,
        y_true: pd.Series,
        predictions: dict[str, pd.Series],
        cv_summary: dict[str, dict] | None = None,
    ) -> pd.DataFrame:
        if not predictions:
            raise ValueError("No model predictions were provided for evaluation.")

        rows: list[dict] = []

        for model_name, y_pred in predictions.items():
            if self.problem_type == "classification":
                metrics = self._score_classification(y_true, y_pred)
            else:
                metrics = self._score_regression(y_true, y_pred)

            row: dict = {"model": model_name, **metrics}

            if cv_summary and model_name in cv_summary:
                row["cv_best_score"] = cv_summary[model_name]["best_score"]
                row["best_params"] = cv_summary[model_name]["best_params"]

            rows.append(row)

        leaderboard = pd.DataFrame(rows)
        sort_col = "f1_weighted" if self.problem_type == "classification" else "r2"
        leaderboard = leaderboard.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        return leaderboard

    def get_winner(self, leaderboard: pd.DataFrame) -> str:
        if leaderboard.empty:
            raise ValueError("Leaderboard is empty; cannot select a winning model.")
        return str(leaderboard.iloc[0]["model"])