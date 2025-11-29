from pathlib import Path
import pandas as pd
from src.data_load import load_data
from src.features import finalize_features
from src.model_mlp import (
    time_split, build_preprocessor, build_mlp_pipeline,
    build_baseline_pipeline, tune_mlp, evaluate_predictions
)
from src.evaluate import plot_predictions, plot_residuals

DATA_PATH = "data/raw/sales.csv"   # Update with your actual file name
DATE_COL = "date"
TARGET_COL = "sales"

def run():
    df = load_data(DATA_PATH, DATE_COL)
    df = finalize_features(df, DATE_COL, TARGET_COL)

    train, val, test = time_split(df, TARGET_COL)
    preprocessor, feature_cols = build_preprocessor(df, TARGET_COL)

    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_val, y_val = val[feature_cols], val[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    baseline_pipe = build_baseline_pipeline(preprocessor)
    baseline_pipe.fit(X_train, y_train)
    baseline_val_pred = baseline_pipe.predict(X_val)
    print("Baseline:", evaluate_predictions(y_val, baseline_val_pred))

    mlp_pipe = build_mlp_pipeline(preprocessor)
    grid = tune_mlp(mlp_pipe, X_train, y_train)
    print("Best MLP params:", grid.best_params_)

    best_model = grid.best_estimator_
    best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_pred = best_model.predict(X_test)
    print("Final Test:", evaluate_predictions(y_test, test_pred))

    Path("figures").mkdir(exist_ok=True)
    plot_predictions(test, DATE_COL, y_test, test_pred, "figures/pred_vs_actual.png")
    plot_residuals(y_test, test_pred, "figures/residuals.png")

if __name__ == "__main__":
    run()
