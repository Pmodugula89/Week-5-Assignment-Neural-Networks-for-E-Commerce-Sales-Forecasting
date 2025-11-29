import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_predictions(df, date_col, y_true, y_pred, out_path):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df[date_col], y_true, label="Actual", color="black")
    ax.plot(df[date_col], y_pred, label="Predicted", color="blue", alpha=0.8)
    ax.set_title("Predicted vs Actual Sales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)

def plot_residuals(y_true, y_pred, out_path):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    ax[0].hist(residuals, bins=30, color="gray")
    ax[0].set_title("Residual Distribution")
    ax[1].scatter(y_pred, residuals, s=10, alpha=0.6)
    ax[1].axhline(0, color="red", linestyle="--")
    ax[1].set_title("Residuals vs Fitted")
    fig.tight_layout()
    fig.savefig(out_path)
