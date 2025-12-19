import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--tracking_uri", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.target_col = args.target_col.strip()

    # --- Validasi file ada ---
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

    # --- MLflow setup ---
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # PENTING: start_run dulu, baru autolog (biar nggak bentrok run ID dari MLflow Project)
    with mlflow.start_run(run_name="RF_baseline"):
        mlflow.sklearn.autolog(
            log_models=True,
            log_input_examples=True,
            log_model_signatures=True
        )

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        rf.fit(X_train, y_train)

        # boleh hitung buat print, tapi JANGAN mlflow.log_* manual
        acc = rf.score(X_test, y_test)
        print(f"[OK] Training selesai. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
