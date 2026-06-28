from pathlib import Path
import pandas as pd

class DataLoader:

    def load_csv(self, path: str) -> pd.DataFrame:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

        if path.suffix != ".csv":
            raise ValueError("Only CSV files are supported.")

        return pd.read_csv(path)