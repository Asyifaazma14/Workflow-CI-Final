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

    # NORMALISASI NAMA KOLOM
    df.columns = df.columns.str.strip()

    # NORMALISASI TARGET
    target_clean = args.target_col.strip()

    if target_clean not in df.columns:
        raise ValueError(
            f"Target col '{target_clean}' tidak ada.\n"
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target_clean])
    y = df[target_clean]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mlflow.set_experiment(args.experiment_name)

    # WAJIB: autolog (sesuai mentor)
    mlflow.sklearn.autolog(log_models=True)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    _ = model.score(X_test, y_test)

    print("Training selesai & tersimpan di MLflow")


if __name__ == "__main__":
    main()
