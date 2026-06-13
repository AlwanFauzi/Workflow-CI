"""
modelling.py (Workflow-CI / MLProject)

Skrip retraining model House Prices yang dijalankan otomatis oleh GitHub Actions
melalui `mlflow run MLProject`. Menggunakan MLflow autolog dan menyimpan model
ke MLflow tracking lokal (./mlruns) sehingga dapat diambil artefaknya dan
di-build menjadi Docker image (mlflow models build-docker).
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET_COL = "SalePrice"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../namadataset_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=str, default="None")
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    return parser.parse_args()


def load_dataset(data_dir):
    train_df = pd.read_csv(os.path.join(data_dir, "train_preprocessed.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_preprocessed.csv"))

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test


def main():
    args = parse_args()
    max_depth = None if args.max_depth in ("None", "none", "") else int(args.max_depth)

    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_dataset(args.data_dir)

    # Buka/lanjutkan run secara eksplisit agar model & metrik (lewat autolog)
    # tercatat di run yang sama, lalu run_id-nya dicetak supaya workflow CI
    # bisa mengambil run yang BENAR-BENAR memiliki artefak `model`.
    with mlflow.start_run() as run:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_r2", r2)

        print(f"Test MAE : {mae:.5f}")
        print(f"Test RMSE: {rmse:.5f}")
        print(f"Test R2  : {r2:.5f}")
        print(f"MLFLOW_RUN_ID={run.info.run_id}")


if __name__ == "__main__":
    main()
