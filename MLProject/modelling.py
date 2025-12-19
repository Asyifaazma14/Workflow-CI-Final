import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--target-col", type=str, required=True)
    p.add_argument("--experiment-name", type=str, required=True)
    p.add_argument("--tracking-uri", type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()

    # bersihin target_col dari quote + spasi aneh
    target = args.target_col.strip().strip('"').strip("'")
    target = " ".join(target.split())

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df.columns = df.columns.str.strip()

    if target not in df.columns:
        raise ValueError(f"Target col '{target}' tidak ada. Kolom: {df.columns.tolist()}")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    mlflow.sklearn.autolog(log_models=True)

    params = {"n_estimators": 200, "max_depth": 10}

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42
    )
    model.fit(X_train, y_train)

    _ = model.score(X_test, y_test)
    print("[OK] Training selesai. Autolog simpan metrics & model.")

if __name__ == "__main__":
    main()
