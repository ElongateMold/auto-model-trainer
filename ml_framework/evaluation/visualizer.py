from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve
from sklearn.preprocessing import label_binarize


class EvaluationVisualizer:
    def __init__(self, problem_type: str, save_dir: Path | None = None, dpi: int = 300):
        if problem_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported problem_type: '{problem_type}'")
        self.problem_type = problem_type
        self.save_dir = save_dir
        self.dpi = dpi

    def plot_all(
        self,
        models: dict[str, BaseEstimator],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        predictions: dict[str, pd.Series],
    ) -> None:
        if not predictions:
            raise ValueError("No predictions provided to visualize.")

        if self.problem_type == "classification":
            self._plot_normalized_confusion_matrices(y_test, predictions)
            self._plot_roc_and_pr_curves(models, X_test, y_test)
        else:
            self._plot_regression_diagnostics(y_test, predictions)

        self._plot_learning_curves(models, X_train, y_train)

    # --- CLASSIFICATION DIAGNOSTICS ---

    def _plot_normalized_confusion_matrices(
        self, y_true: pd.Series, predictions: dict[str, pd.Series]
    ) -> None:
        n_models = len(predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        axes_list = [axes] if n_models == 1 else list(np.ravel(axes))

        for ax, (model_name, y_pred) in zip(axes_list, predictions.items()):
            ConfusionMatrixDisplay.from_predictions(
                y_true=y_true,
                y_pred=y_pred,
                normalize="true",
                values_format=".2f",
                ax=ax,
                cmap="Blues",
                colorbar=False,
            )
            ax.set_title(f"{model_name}\nNormalized Confusion Matrix")

        plt.tight_layout()
        self._finalize_figure(fig, "confusion_matrices_normalized.png")

    def _get_class_score_matrix(
        self, model: BaseEstimator, X_test: pd.DataFrame, classes: np.ndarray
    ) -> np.ndarray:
        """
        Extracts a (n_samples, n_classes) matrix of continuous confidence scores.
        Supports predict_proba(), decision_function(), and hard predict() fallbacks (e.g., LVQ1/DANN).
        """
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(X_test), dtype=float)

        if hasattr(model, "decision_function"):
            raw_scores = np.asarray(model.decision_function(X_test), dtype=float)
            if raw_scores.ndim == 1:
                prob_pos = 1.0 / (1.0 + np.exp(-raw_scores))
                return np.column_stack([1.0 - prob_pos, prob_pos])
            return raw_scores

        # Fallback for prototype classifiers (LVQ1 / DANN) that only emit discrete labels
        hard_preds = np.asarray(model.predict(X_test))
        binarized = np.asarray(label_binarize(hard_preds, classes=classes), dtype=float)
        if len(classes) == 2:
            return np.column_stack([1.0 - binarized[:, 0], binarized[:, 0]])
        return binarized

    def _plot_roc_and_pr_curves(
        self,
        models: dict[str, BaseEstimator],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        classes = np.sort(y_test.unique())
        n_classes = len(classes)

        y_bin = np.asarray(label_binarize(y_test, classes=classes), dtype=int)
        if n_classes == 2:
            y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])

        n_models = len(models)
        fig, axes = plt.subplots(2, n_models, figsize=(6.5 * n_models, 11))
        axes_grid = axes.reshape(2, n_models) if n_models == 1 else axes

        for col_idx, (model_name, model) in enumerate(models.items()):
            score_matrix = self._get_class_score_matrix(model, X_test, classes)
            ax_roc = axes_grid[0, col_idx]
            ax_pr = axes_grid[1, col_idx]

            # Per-class One-vs-Rest ROC and Precision-Recall curves
            for class_idx, class_label in enumerate(classes):
                y_true_c = y_bin[:, class_idx]
                y_score_c = score_matrix[:, class_idx]

                fpr, tpr, _ = roc_curve(y_true_c, y_score_c)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, lw=1.8, label=f"Class {class_label} (AUC = {roc_auc:.2f})")

                precision, recall, _ = precision_recall_curve(y_true_c, y_score_c)
                ap_score = average_precision_score(y_true_c, y_score_c)
                ax_pr.plot(recall, precision, lw=1.8, label=f"Class {class_label} (AP = {ap_score:.2f})")

            # Micro-average curves across all classes
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), score_matrix.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            ax_roc.plot(
                fpr_micro,
                tpr_micro,
                linestyle="--",
                color="black",
                lw=2,
                label=f"Micro-avg (AUC = {auc_micro:.2f})",
            )
            ax_roc.plot([0, 1], [0, 1], "k:", lw=1, label="Random Chance")
            ax_roc.set_title(f"{model_name}: Multiclass ROC Curve")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right", fontsize=8)

            prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), score_matrix.ravel())
            ap_micro = average_precision_score(y_bin, score_matrix, average="micro")
            ax_pr.plot(
                rec_micro,
                prec_micro,
                linestyle="--",
                color="black",
                lw=2,
                label=f"Micro-avg (AP = {ap_micro:.2f})",
            )
            ax_pr.set_title(f"{model_name}: Precision-Recall Curve")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.legend(loc="lower left", fontsize=8)

        plt.tight_layout()
        self._finalize_figure(fig, "roc_and_precision_recall_curves.png")

    # --- REGRESSION DIAGNOSTICS ---

    def _plot_regression_diagnostics(
        self, y_true: pd.Series, predictions: dict[str, pd.Series]
    ) -> None:
        n_models = len(predictions)
        fig, axes = plt.subplots(3, n_models, figsize=(6 * n_models, 14))
        axes_grid = axes.reshape(3, n_models) if n_models == 1 else axes

        y_true_arr = y_true.to_numpy(dtype=float)
        min_val = float(np.min(y_true_arr))
        max_val = float(np.max(y_true_arr))

        for col_idx, (model_name, y_pred) in enumerate(predictions.items()):
            y_pred_arr = y_pred.to_numpy(dtype=float)
            residuals = y_true_arr - y_pred_arr

            # Row 0: Actual vs. Predicted with 95% Confidence Band
            ax_fit = axes_grid[0, col_idx]
            sns.regplot(
                x=y_true_arr,
                y=y_pred_arr,
                ci=95,
                ax=ax_fit,
                scatter_kws={"alpha": 0.65, "edgecolor": "k"},
                line_kws={"color": "navy", "label": "OLS Fit (95% CI)"},
            )
            ax_fit.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                linewidth=1.8,
                label="Ideal (y = x)",
            )
            ax_fit.set_title(f"{model_name}: Actual vs. Predicted (95% CI)")
            ax_fit.set_xlabel("Actual Values")
            ax_fit.set_ylabel("Predicted Values")
            ax_fit.legend()

            # Row 1: Residuals vs. Predicted
            ax_res = axes_grid[1, col_idx]
            ax_res.scatter(y_pred_arr, residuals, alpha=0.65, color="darkorange", edgecolors="k")
            ax_res.axhline(0.0, color="red", linestyle="--", linewidth=1.8, label="Zero Error")
            ax_res.set_title(f"{model_name}: Residuals vs. Predicted")
            ax_res.set_xlabel("Predicted Values")
            ax_res.set_ylabel("Residuals (Actual - Predicted)")
            ax_res.legend()

            # Row 2: Residual Distribution (Histogram + KDE)
            ax_dist = axes_grid[2, col_idx]
            sns.histplot(
                residuals,
                kde=True,
                ax=ax_dist,
                color="teal",
                edgecolor="black",
                alpha=0.6,
            )
            ax_dist.axvline(0.0, color="red", linestyle="--", linewidth=1.8, label="Zero Mean")
            ax_dist.set_title(f"{model_name}: Residual Distribution")
            ax_dist.set_xlabel("Residual Error")
            ax_dist.set_ylabel("Frequency")
            ax_dist.legend()

        plt.tight_layout()
        self._finalize_figure(fig, "regression_diagnostics.png")

    # --- SHARED LEARNING CURVES (ERROR VS. SAMPLE SIZE) ---

    def _plot_learning_curves(
        self,
        models: dict[str, BaseEstimator],
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        axes_list = [axes] if n_models == 1 else list(np.ravel(axes))

        if self.problem_type == "classification":
            cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "accuracy"
            y_label = "Classification Error (1 - Accuracy)"
        else:
            cv_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "neg_root_mean_squared_error"
            y_label = "RMSE Error"

        for ax, (model_name, model) in zip(axes_list, models.items()):
            train_sizes, train_scores, val_scores = learning_curve(
                estimator=clone(model),
                X=X_train,
                y=y_train,
                cv=cv_splitter,
                scoring=scoring,
                train_sizes=np.linspace(0.2, 1.0, 5),
                n_jobs=-1,
            )

            # Convert scikit-learn scores into positive error values
            if self.problem_type == "classification":
                train_errors = 1.0 - train_scores
                val_errors = 1.0 - val_scores
            else:
                train_errors = -train_scores
                val_errors = -val_scores

            train_mean = np.mean(train_errors, axis=1)
            train_std = np.std(train_errors, axis=1)
            val_mean = np.mean(val_errors, axis=1)
            val_std = np.std(val_errors, axis=1)

            ax.plot(train_sizes, train_mean, "o-", color="royalblue", label="Training Error")
            ax.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.15,
                color="royalblue",
            )

            ax.plot(train_sizes, val_mean, "o-", color="crimson", label="Validation Error (CV)")
            ax.fill_between(
                train_sizes,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.15,
                color="crimson",
            )

            ax.set_title(f"{model_name}: Learning Curve")
            ax.set_xlabel("Training Sample Size")
            ax.set_ylabel(y_label)
            ax.legend(loc="best")

        plt.tight_layout()
        self._finalize_figure(fig, "learning_curves.png")

    def _finalize_figure(self, fig, filename: str) -> None:
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(self.save_dir / filename, dpi=self.dpi)
            plt.close(fig)
        else:
            plt.show()