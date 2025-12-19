import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline RandomForest with MLflow autolog.")
    parser.add_argument("--data_path", type=str, required=True, help="Path ke CSV data bersih (hasil preprocessing).")
    parser.add_argument("--target_col", type=str, required=True, help="Nama kolom target.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Nama experiment MLflow.")
    parser.add_argument("--tracking_uri", type=str, required=True, help="Tracking URI MLflow (contoh: file:./mlruns).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validasi file dataset
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path}")

    # Load data
    df = pd.read_csv(args.data_path)
    df.columns = df.columns.str.strip()

    # Validasi target
    if args.target_col not in df.columns:
        raise ValueError(
            f"Target col '{args.target_col}' tidak ada. Kolom tersedia: {df.columns.tolist()}"
        )

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Setup MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Autolog (sesuai arahan mentor: tanpa logging manual)
    mlflow.sklearn.autolog(log_models=True)

    # Baseline params (1 kombinasi saja)
    params = {"n_estimators": 200, "max_depth": 10, "random_state": 42}

    with mlflow.start_run(run_name=f"RF_baseline_{params['n_estimators']}_{params['max_depth']}"):
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)

        # Trigger autolog metrics
        _ = model.score(X_test, y_test)

    print("[OK] Training selesai. Artefak & metrik tersimpan di MLflow.")


if __name__ == "__main__":
    main()
