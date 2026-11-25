from abc import ABC, abstractmethod
from pathlib import Path
import warnings
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype


class DataReader(ABC):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._validate_file()

    def _validate_file(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: '{self.path}'")
        if not self.path.is_file():
            raise ValueError(f"Path is not a valid file: '{self.path}'")

    @abstractmethod
    def read(self) -> pd.DataFrame:
        pass


class DelimitedTextReader(DataReader):
    def __init__(self, path: str | Path, sep: str):
        super().__init__(path)
        self.sep = sep

    def read(self) -> pd.DataFrame:
        for encoding in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(self.path, sep=self.sep, encoding=encoding)
                # If a .csv or .tsv parses as 1 column, auto-sniff the true delimiter (\t, ;, |)
                if df.shape[1] == 1:
                    sniffed_df = pd.read_csv(
                        self.path, sep=None, engine="python", encoding=encoding
                    )
                    if sniffed_df.shape[1] > 1:
                        return sniffed_df
                return df
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(
            "utf-8/latin-1", b"", 0, 1, f"Unable to decode '{self.path}' with utf-8 or latin-1."
        )


class CSVReader(DelimitedTextReader):
    def __init__(self, path: str | Path):
        super().__init__(path, sep=",")


class TSVReader(DelimitedTextReader):
    def __init__(self, path: str | Path):
        super().__init__(path, sep="\t")


class ParquetReader(DataReader):
    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)


def get_reader(path: str | Path) -> DataReader:
    file_path = Path(path)
    extension = file_path.suffix.lower()

    if extension == ".csv":
        return CSVReader(file_path)
    elif extension == ".tsv":
        return TSVReader(file_path)
    elif extension == ".parquet":
        return ParquetReader(file_path)
    else:
        raise ValueError(f"Unsupported file format: '{extension}'")


class DatasetLoader:
    COMMON_TARGET_NAMES = {
        "target", "label", "class", "y", "output", "outcome",
        "prediction", "churn", "price", "diagnosis", "status",
        "species", "survived", "salary", "sales",
    }

    def __init__(self, max_target_classes: int = 15):
        self.max_target_classes = max_target_classes

    def load(
        self,
        path: str | Path,
        user_target: str | None = None,
        force_unsupervised: bool = False,
    ) -> tuple[pd.DataFrame, str | None, str]:
        df = get_reader(path).read()
        if df.empty:
            raise ValueError(f"Loaded dataset '{path}' is empty.")

        if force_unsupervised:
            return df, None, "unsupervised"

        target_col, mode = self.detect_supervision(df, user_target)
        return df, target_col, mode

    def detect_supervision(
        self, df: pd.DataFrame, user_target: str | None = None
    ) -> tuple[str | None, str]:
        # 1. Explicit user target always takes priority
        if user_target is not None:
            if user_target not in df.columns:
                raise ValueError(f"Specified target column '{user_target}' does not exist.")
            return user_target, "supervised"

        # 2. Search for standard target column names
        for col in df.columns:
            if col.strip().lower() in self.COMMON_TARGET_NAMES:
                return col, "supervised"

        # 3. Inspect the rightmost non-ID column for a clear categorical/discrete label
        non_id_cols = [
            col for col in df.columns
            if not (col.strip().lower() == "id" or col.strip().lower().endswith("_id"))
        ]

        if non_id_cols:
            last_col = non_id_cols[-1]
            series = df[last_col].dropna()
            unique_count = series.nunique()

            is_categorical_label = (
                (is_bool_dtype(series) or not is_numeric_dtype(series))
                and 2 <= unique_count <= self.max_target_classes
            )
            is_discrete_int_label = (
                is_integer_dtype(series)
                and 2 <= unique_count <= self.max_target_classes
            )

            if is_categorical_label or is_discrete_int_label:
                return last_col, "supervised"

        # 4. No evident target candidate -> emit warning and route to Unsupervised K-Means
        warnings.warn(
            "No target column was specified and no obvious label column was detected. "
            "Routing execution automatically to the Unsupervised K-Means Engine.",
            UserWarning,
            stacklevel=2,
        )
        return None, "unsupervised"