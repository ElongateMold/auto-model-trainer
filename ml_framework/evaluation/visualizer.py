from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


class EvaluationVisualizer:
    def __init__(self, problem_type: str, save_dir: Path | None = None):
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem_type: '{problem_type}'")
        self.problem_type = problem_type
        self.save_dir = save_dir

    def plot_evaluation(self, y_true: pd.Series, predictions: dict[str, pd.Series]) -> None:
        if not predictions:
            raise ValueError("No predictions provided to visualize.")

        if self.problem_type == "classification":
            self._plot_confusion_matrices(y_true, predictions)
        else:
            self._plot_regression_diagnostics(y_true, predictions)

    def _plot_confusion_matrices(
        self, y_true: pd.Series, predictions: dict[str, pd.Series]
    ) -> None:
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        axes_list = [axes] if n_models == 1 else list(np.ravel(axes))

        for ax, (model_name, y_pred) in zip(axes_list, predictions.items()):
            ConfusionMatrixDisplay.from_predictions(
                y_true=y_true,
                y_pred=y_pred,
                ax=ax,
                cmap="Blues",
                colorbar=False,
            )
            ax.set_title(f"{model_name}\nConfusion Matrix")

        plt.tight_layout()
        self._finalize_figure(fig, "confusion_matrices.png")

    def _plot_regression_diagnostics(
        self, y_true: pd.Series, predictions: dict[str, pd.Series]
    ) -> None:
        n_models = len(predictions)
        fig, axes = plt.subplots(2, n_models, figsize=(6 * n_models, 10))
        axes_grid = axes.reshape(2, n_models) if n_models == 1 else axes

        y_true_arr = y_true.to_numpy(dtype=float)
        min_val = float(np.min(y_true_arr))
        max_val = float(np.max(y_true_arr))

        for col_idx, (model_name, y_pred) in enumerate(predictions.items()):
            y_pred_arr = y_pred.to_numpy(dtype=float)
            residuals = y_true_arr - y_pred_arr

            # Row 0: Actual vs. Predicted
            ax_top = axes_grid[0, col_idx]
            ax_top.scatter(y_true_arr, y_pred_arr, alpha=0.65, edgecolors="k")
            ax_top.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = x)")
            ax_top.set_title(f"{model_name}: Actual vs. Predicted")
            ax_top.set_xlabel("Actual Values")
            ax_top.set_ylabel("Predicted Values")
            ax_top.legend()

            # Row 1: Residuals vs. Predicted
            ax_bot = axes_grid[1, col_idx]
            ax_bot.scatter(y_pred_arr, residuals, alpha=0.65, color="darkorange", edgecolors="k")
            ax_bot.axhline(0.0, color="red", linestyle="--", linewidth=2, label="Zero Error")
            ax_bot.set_title(f"{model_name}: Residual Plot")
            ax_bot.set_xlabel("Predicted Values")
            ax_bot.set_ylabel("Residuals (Actual - Predicted)")
            ax_bot.legend()

        plt.tight_layout()
        self._finalize_figure(fig, "regression_diagnostics.png")

    def _finalize_figure(self, fig, filename: str) -> None:
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.save_dir / filename, dpi=150)
            plt.close(fig)
        else:
            plt.show()