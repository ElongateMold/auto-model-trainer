import pandas as pd


class ModelSelector:
    def __init__(
        self,
        df: pd.DataFrame,
        diagnostic_report: dict,
        outlier_pct_threshold: float = 5.0,
        non_linear_gap_threshold: float = 0.15,
    ):
        self.df = df.copy()
        self.report = diagnostic_report
        self.target_col: str = str(diagnostic_report["target"])
        self.problem_type: str = str(diagnostic_report["problem_type"])
        self.outlier_pct_threshold = outlier_pct_threshold
        self.non_linear_gap_threshold = non_linear_gap_threshold

    def get_columns_to_drop(self) -> dict[str, list[str]]:
        # 1. Zero or near-zero variance columns
        low_var = [
            col for col in self.report["low_variance_columns"]
            if col != self.target_col
        ]

        # 2. High-cardinality text columns
        card_df: pd.DataFrame = self.report["categorical_cardinality"]
        high_card: list[str] = []
        if not card_df.empty:
            mask = card_df["high_cardinality"].astype(bool)
            flagged = card_df[mask]["column"].tolist()
            high_card = [col for col in flagged if col != self.target_col]

        # 3. Identifier columns
        id_cols = [
            col for col in self.df.columns
            if col != self.target_col
            and (col.strip().lower() == "id" or col.strip().lower().endswith("_id"))
        ]

        # 4. Redundant collinear columns (keep first feature, drop second)
        collinear_to_drop: list[str] = []
        already_dropped = set(low_var + high_card + id_cols)

        for pair in self.report["high_multicollinearity"]:
            feat_1 = pair["feature_1"]
            feat_2 = pair["feature_2"]

            if feat_1 == self.target_col or feat_2 == self.target_col:
                continue
            if feat_1 in already_dropped or feat_2 in already_dropped:
                continue

            collinear_to_drop.append(feat_2)
            already_dropped.add(feat_2)

        return {
            "low_variance": low_var,
            "high_cardinality": high_card,
            "id_columns": id_cols,
            "multicollinear": collinear_to_drop,
        }
    # prune_features() is made for removing features that are not useful for modeling
    def prune_features(self) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        dropped_summary = self.get_columns_to_drop()
        all_to_drop = set(
            dropped_summary["low_variance"]
            + dropped_summary["high_cardinality"]
            + dropped_summary["id_columns"]
            + dropped_summary["multicollinear"]
        )
        clean_df = self.df.drop(columns=list(all_to_drop), errors="ignore")
        return clean_df, dropped_summary

    def select_model_family(self, surviving_cols: list[str]) -> dict:
        reasons: list[str] = []

        # Check outlier percentage in surviving numeric columns
        outliers_df: pd.DataFrame = self.report["outliers_iqr"]
        if not outliers_df.empty:
            active_outliers = outliers_df[outliers_df["column"].isin(surviving_cols)]
            heavy_cols = active_outliers[
                active_outliers["outlier_pct"] > self.outlier_pct_threshold
            ]["column"].tolist()

            if heavy_cols:
                reasons.append(
                    f"High outlier rate (above {self.outlier_pct_threshold}%) in {heavy_cols}; "
                    "tree splits isolate extreme values better than linear equations."
                )

        # Compare Spearman vs. Pearson to detect non-linear curves
        pearson_df: pd.DataFrame = self.report["matrices"]["pearson_correlation"]
        spearman_df: pd.DataFrame = self.report["matrices"]["spearman_correlation"]
        numeric_surviving = [c for c in pearson_df.columns if c in surviving_cols]

        if len(numeric_surviving) >= 2:
            diff_matrix = (
                spearman_df.loc[numeric_surviving, numeric_surviving].abs()
                - pearson_df.loc[numeric_surviving, numeric_surviving].abs()
            )
            stacked_diffs = diff_matrix.stack()
            max_gap = float(stacked_diffs.max()) if not stacked_diffs.empty else 0.0

            if max_gap > self.non_linear_gap_threshold:
                reasons.append(
                    f"Spearman correlation exceeds Pearson by {max_gap:.2f}, indicating "
                    "non-linear relationships better captured by tree-based models."
                )

        # Final selection of model family based on reasons collected
        if reasons:
            family = "tree_based"
            models = (
                ["RandomForestClassifier", "GradientBoostingClassifier"]
                if self.problem_type == "classification"
                else ["RandomForestRegressor", "GradientBoostingRegressor"]
            )
        else:
            family = "linear"
            reasons.append(
                "Low outlier rates and linear feature relationships favor fast, "
                "interpretable linear models."
            )
            models = (
                ["LogisticRegression", "RidgeClassifier"]
                if self.problem_type == "classification"
                else ["LinearRegression", "Ridge"]
            )

        return {
            "model_family": family,
            "recommended_models": models,
            "reasons": reasons,
        }

    def run(self) -> dict:
        pruned_df, dropped_summary = self.prune_features()
        surviving_features = [c for c in pruned_df.columns if c != self.target_col]
        model_decision = self.select_model_family(surviving_features)

        return {
            "pruned_df": pruned_df,
            "target": self.target_col,
            "problem_type": self.problem_type,
            "surviving_features": surviving_features,
            "dropped_columns": dropped_summary,
            "model_family": model_decision["model_family"],
            "recommended_models": model_decision["recommended_models"],
            "reasons": model_decision["reasons"],
        }