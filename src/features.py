import pandas as pd

def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    dt = df[date_col]
    df["dow"] = dt.dt.dayofweek
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, target_col: str, lags=(1,7)) -> pd.DataFrame:
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, target_col: str, windows=(7,14)) -> pd.DataFrame:
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(window=w).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].rolling(window=w).std()
    return df

def finalize_features(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    df = add_calendar_features(df, date_col)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = df.dropna()
    return df
