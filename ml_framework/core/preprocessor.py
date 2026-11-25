import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self, scale_numeric: bool = True):
        self.scale_numeric = scale_numeric
        self.pipeline: ColumnTransformer | None = None
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = []

    def _build_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        self.numeric_cols = X.select_dtypes(
            include=["number"], exclude=["bool"]
        ).columns.tolist()
        self.categorical_cols = X.select_dtypes(
            include=["object", "string", "category", "bool"]
        ).columns.tolist()

        numeric_steps: list = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_transformer = Pipeline(steps=numeric_steps)

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_cols),
                ("cat", categorical_transformer, self.categorical_cols),
            ],
            remainder="drop",
        )

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.pipeline = self._build_pipeline(X_train)
        transformed_array = self.pipeline.fit_transform(X_train)
        feature_names = self.pipeline.get_feature_names_out()
        return pd.DataFrame(transformed_array, columns=feature_names, index=X_train.index)

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Call fit_transform() on training data before transform().")
        transformed_array = self.pipeline.transform(X_test)
        feature_names = self.pipeline.get_feature_names_out()
        return pd.DataFrame(transformed_array, columns=feature_names, index=X_test.index)