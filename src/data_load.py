import pandas as pd
from pathlib import Path

def load_data(path: str, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df
