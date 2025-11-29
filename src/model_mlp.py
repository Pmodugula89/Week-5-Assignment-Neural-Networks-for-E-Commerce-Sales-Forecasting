import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def time_split(df: pd.DataFrame, target_col: str, train_frac=0.7, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test

def build_preprocessor(df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c not in [target_col, "date"]]
    cat_cols = [c for c in feature_cols if df[c].nunique() < 10]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])
    return preprocessor, feature_cols

def build_mlp_pipeline(preprocessor):
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42
    )
    return Pipeline([("prep", preprocessor), ("model", mlp)])

def build_baseline_pipeline(preprocessor):
    ridge = Ridge(alpha=1.0)
    return Pipeline([("prep", preprocessor), ("model", ridge)])

def tune_mlp(pipe, X_train, y_train):
    tscv = TimeSeriesSplit(n_splits=4)
    param_grid = {
        "model__hidden_layer_sizes": [(64,), (64,32)],
        "model__alpha": [0.0001, 0.001],
        "model__learning_rate_init": [0.001, 0.0005],
        "model__max_iter": [400, 600]
    }
    grid = GridSearchCV(pipe, param_grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

def evaluate_predictions(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred)
    }
