from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import seaborn as sns


class EDAAnalyzer:
    COMMON_TARGET_NAMES = {
        "target", "label", "class", "y", "output",
        "outcome", "prediction", "churn", "price", "diagnosis", "status",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        max_classes: int = 15,
        unsupervised: bool = False,
    ):
        self.df = df
        self.max_classes = max_classes
        self.target_col = self._resolve_target(target_col)
        self.unsupervised = unsupervised

    # --- TARGET & PROBLEM TYPE DETECTION (FROM PREVIOUS STEP) ---

    def _resolve_target(self, target_col: str | None) -> str:
        if target_col is not None:
            if target_col not in self.df.columns:
                raise ValueError(f"Target column '{target_col}' does not exist.")
            return target_col

        for col in self.df.columns:
            if col.strip().lower() in self.COMMON_TARGET_NAMES:
                return col

        for col in reversed(self.df.columns):
            if not self._is_id_column(col):
                return col

        return str(self.df.columns[-1])

    def _is_id_column(self, col: str) -> bool:
        clean_name = col.strip().lower()
        return clean_name == "id" or clean_name.endswith("_id")

    def infer_problem_type(self) -> str:
        if self.target_col is None:
            raise ValueError("Cannot infer supervised problem type in unsupervised mode.")
        target_series = self.df[self.target_col].dropna()
        if target_series.empty:
            raise ValueError(f"Target column '{self.target_col}' has only missing values.")

        if is_bool_dtype(target_series) or not is_numeric_dtype(target_series):
            return "classification"

        if (target_series % 1 != 0).any():
            return "regression"

        if target_series.nunique() <= self.max_classes:
            return "classification"

        return "regression"

    # --- 1. DATA QUALITY DIAGNOSTICS ---

    def check_missing_values(self) -> pd.DataFrame:
        missing_count = self.df.isnull().sum()
        missing_pct = (self.df.isnull().mean() * 100).round(2)

        report = pd.DataFrame({
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        })
        return report[report["missing_count"] > 0]

    def check_categorical_cardinality(self) -> pd.DataFrame:
        cat_cols = self.df.select_dtypes(include=["object", "string", "category"]).columns
        rows = []

        for col in cat_cols:
            unique_count = int(self.df[col].nunique(dropna=True))
            unique_ratio = round(unique_count / max(len(self.df), 1), 4)
            rows.append({
                "column": col,
                "unique_values": unique_count,
                "unique_ratio": unique_ratio,
                "high_cardinality": unique_ratio > 0.5,
            })

        return pd.DataFrame(rows)

    def detect_outliers_iqr(self) -> pd.DataFrame:
        numeric_df = self.df.select_dtypes(include=["number"], exclude=["bool"])
        rows = []

        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())

            rows.append({
                "column": col,
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4),
                "outlier_count": outlier_count,
                "outlier_pct": round((outlier_count / len(series)) * 100, 2),
            })

        return pd.DataFrame(rows)

    # --- 2. DISPERSION & BIVARIATE RELATIONSHIPS ---

    def find_low_variance_columns(self, variance_threshold: float = 1e-5) -> list[str]:
        low_var_cols = []

        for col in self.df.columns:
            # Constant columns (0 or 1 unique value) have zero variance regardless of dtype
            if self.df[col].nunique(dropna=True) <= 1:
                low_var_cols.append(col)
            elif is_numeric_dtype(self.df[col]) and not is_bool_dtype(self.df[col]):
                if float(self.df[col].var()) <= variance_threshold:
                    low_var_cols.append(col)

        return low_var_cols

    def find_high_multicollinearity(self, threshold: float = 0.85) -> list[dict]:
        numeric_df = self.df.select_dtypes(include=["number"], exclude=["bool"])
        corr_matrix = numeric_df.corr(method="pearson").abs()

        cols = corr_matrix.columns
        high_corr_pairs = []

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_value = corr_matrix.iloc[i, j]
                if pd.notna(corr_value) and corr_value > threshold:
                    high_corr_pairs.append({
                        "feature_1": cols[i],
                        "feature_2": cols[j],
                        "abs_correlation": round(float(corr_value), 4),
                    })

        return high_corr_pairs

    # --- 3. COVARIANCE, CORRELATION & ANNOTATED HEATMAPS ---

    def compute_matrices(self) -> dict[str, pd.DataFrame]:
        numeric_df = self.df.select_dtypes(include=["number"], exclude=["bool"])
        return {
            "covariance": numeric_df.cov(),
            "pearson_correlation": numeric_df.corr(method="pearson"),
            "spearman_correlation": numeric_df.corr(method="spearman"),
        }

    def render_correlation_heatmaps(self, save_dir: Path | None = None) -> None:
        matrices = self.compute_matrices()
        pearson_corr = matrices["pearson_correlation"]
        spearman_corr = matrices["spearman_correlation"]

        if pearson_corr.empty:
            print("No numeric columns available to plot correlation heatmaps.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(
            pearson_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=axes[0],
        )
        axes[0].set_title("Pearson Correlation (Linear Relationships)")

        sns.heatmap(
            spearman_corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=axes[1],
        )
        axes[1].set_title("Spearman Correlation (Monotonic / Non-Linear)")

        plt.tight_layout()

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / "correlation_heatmaps.png"
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

    # --- MASTER AUTOMATED DIAGNOSTIC REPORT ---

    def run_structural_diagnostic(self, save_plots_dir: Path | None = None) -> dict:
        matrices = self.compute_matrices()
        self.render_correlation_heatmaps(save_dir=save_plots_dir)

        return {
            "target": self.target_col,
            "problem_type": "unsupervised" if self.unsupervised else self.infer_problem_type(),
            "missing_values": self.check_missing_values(),
            "categorical_cardinality": self.check_categorical_cardinality(),
            "outliers_iqr": self.detect_outliers_iqr(),
            "low_variance_columns": self.find_low_variance_columns(),
            "high_multicollinearity": self.find_high_multicollinearity(threshold=0.85),
            "matrices": matrices,
        }