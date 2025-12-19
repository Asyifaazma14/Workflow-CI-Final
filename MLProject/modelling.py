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
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df.columns = df.columns.str.strip()

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target col '{args.target_col}' tidak ada. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment(args.experiment_name)

    # WAJIB: autolog (tanpa manual log)
    mlflow.sklearn.autolog(log_models=True)

    params = {"n_estimators": 200, "max_depth": 10}
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42
    )
    model.fit(X_train, y_train)

    # cukup trigger metric supaya terekam (autolog akan log otomatis)
    _ = model.score(X_test, y_test)

    print("[OK] Training selesai. Autolog menyimpan metrics + model ke MLflow.")


if __name__ == "__main__":
    main()
