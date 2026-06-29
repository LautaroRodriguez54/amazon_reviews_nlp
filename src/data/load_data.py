from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Womens Clothing E-Commerce Reviews.csv"


def load_data() -> pd.DataFrame:
    """
    Carga el dataset Women's E-Commerce Clothing Reviews.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en:\n{DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":

    df = load_data()

    print(df.head())