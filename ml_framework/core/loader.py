from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

# Clase base
class DataReader(ABC):
    def __init__(self, path: str | Path):
        self.path = path

    @abstractmethod
    def read(self) -> pd.DataFrame:
        # Every subclass must implement this method to read data from the specified path and return a DataFrame.
        pass

class CSVReader(DataReader):
    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

class ParquetReader(DataReader):
    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

class TSVReader(DataReader):
    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path, sep='\t')

def get_reader(file_path: str | Path) -> DataReader:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()
    if extension == '.csv':
        return CSVReader(file_path)
    elif extension == '.parquet':
        return ParquetReader(file_path)
    elif extension == '.tsv':
        return TSVReader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}")