import pandas as pd


class DataCleaner:
    def __init__(self, df: pd.DataFrame, max_unique_ratio: float = 0.5):
        self.df = df.copy()
        self.max_unique_ratio = max_unique_ratio

    def clean_categories(self, interactive: bool = True) -> pd.DataFrame:
        text_cols = self.df.select_dtypes(include=["object", "string", "category"]).columns

        for col in text_cols:
            if self._is_high_cardinality(self.df[col]):
                continue

            variants_map = self._find_variants(self.df[col])
            if not variants_map:
                continue

            replacement_rules: dict[str, str] = {}

            for normalized_key, raw_variants in variants_map.items():
                if interactive:
                    chosen = self._ask_user_choice(col, raw_variants)
                else:
                    chosen = self._most_frequent(self.df[col], raw_variants)

                for raw_val in raw_variants:
                    if raw_val != chosen:
                        replacement_rules[raw_val] = chosen

            if replacement_rules:
                self.df[col] = self.df[col].astype("object").replace(replacement_rules)

        return self.df

    def _is_high_cardinality(self, series: pd.Series) -> bool:
        total_rows = len(series)
        if total_rows == 0:
            return True
        unique_ratio = series.nunique(dropna=True) / total_rows
        return unique_ratio > self.max_unique_ratio

    def _normalize(self, text: str) -> str:
        cleaned = text.strip().lower().replace("-", " ").replace("_", " ")
        return " ".join(cleaned.split())

    def _find_variants(self, series: pd.Series) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}

        for val in series.dropna().unique():
            text_val = str(val)
            key = self._normalize(text_val)
            grouped.setdefault(key, []).append(text_val)

        return {key: vals for key, vals in grouped.items() if len(vals) > 1}

    def _most_frequent(self, series: pd.Series, raw_variants: list[str]) -> str:
        counts = series.value_counts()
        return max(raw_variants, key=lambda val: counts.get(val, 0))

    def _ask_user_choice(self, col: str, raw_variants: list[str]) -> str:
        print(f"\n[Column '{col}'] Multiple forms detected for the same category:")
        for idx, variant in enumerate(raw_variants, start=1):
            print(f"  {idx}. '{variant}'")
        print("  0. Keep all as-is (skip)")

        while True:
            choice = input("Choose a number to standardize (or type a custom name): ").strip()

            if choice == "0":
                return ""
            if choice.isdigit() and 1 <= int(choice) <= len(raw_variants):
                return raw_variants[int(choice) - 1]
            if choice:
                return choice