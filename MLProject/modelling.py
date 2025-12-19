import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--experiment_name", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df.columns = df.columns.str.strip()

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target col '{args.target_col}' tidak ada. Kolom tersedia: {df.columns.tolist()}"
        )

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # penting: set experiment boleh, tapi tracking uri JANGAN di-set di sini (biar MLflow Project yang ngatur)
    mlflow.set_experiment(args.experiment_name)

    # wajib autolog (sesuai arahan mentor)
    mlflow.sklearn.autolog(log_models=True)

    params = {"n_estimators": 200, "max_depth": 10, "random_state": 42}

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # hitung metric manual boleh (ini masih aman, bukan "manual logging"; cuma hitung)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # log metric manual SEBAIKNYA ga usah kalau mentor super ketat.
    # Tapi kalau boleh, ini membantu:
    # mlflow.log_metric("test_accuracy", acc)

    print(f"[OK] Training selesai. Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
