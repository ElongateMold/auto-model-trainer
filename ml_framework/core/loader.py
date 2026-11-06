from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

# Clase base
class DataReader(ABC):
    def __init__(self, path: Path):
        self.path = path