import pandas as pd

class EDA:
    COMMON_TARGET_NAMES = {
        "target",
        "label",
        "class",
        "y",
        "output",
        "outcome",
        "prediction",
        "churn",
        "price",
        "diagnosis",
        "status",
    }

    def __init__(self, df: pd.DataFrame, target_col: str | None = None):
        self.df = df
        self.target_col = self._resolve_target(target_col)

    def _resolve_target(self, target_col: str | None) -> str:
        # Tier 1: User explicitly specified a target column
        if target_col is not None:
            if target_col not in self.df.columns:
                raise ValueError(f"Target column '{target_col}' does not exist in the DataFrame.")
            return target_col

        # Tier 2: Look for standard target column names
        for col in self.df.columns:
            if col.strip().lower() in self.COMMON_TARGET_NAMES:
                return col

        # Tier 3: Fallback to the rightmost column that is not an ID
        for col in reversed(self.df.columns):
            if not self._is_id_column(col):
                return col

        # Safety net: if all columns look like IDs, return the very last column
        return str(self.df.columns[-1])

    def _is_id_column(self, col: str) -> bool:
        clean_name = col.strip().lower()
        return clean_name == "id" or clean_name.endswith("_id")