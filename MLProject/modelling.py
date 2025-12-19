import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()

    # MLflow Projects sering mengirim argumen underscore: --data_path
    # Tapi biar aman, kita terima juga versi dash: --data-path
    p.add_argument("--data_path", "--data-path", dest="data_path", type=str, required=True)
    p.add_argument("--target_col", "--target-col", dest="target_col", type=str, required=True)
    p.add_argument("--experiment_name", "--experiment-name", dest="experiment_name", type=str, required=True)
    p.add_argument("--tracking_uri", "--tracking-uri", dest="tracking_uri", type=str, required=True)

    return p.parse_args()


def main():
    args = parse_args()

    # bersihin kemungkinan nilai kebawa quote ganda, contoh: '"Sleep Disorder"'
    target_col = str(args.target_col).strip().strip('"').strip("'")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise ValueError(f"Target col '{target_col}' tidak ada. Kolom tersedia: {df.columns.tolist()}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    # Wajib autolog (sesuai arahan mentor kamu)
    mlflow.sklearn.autolog(log_models=True)

    params = {"n_estimators": 200, "max_depth": 10, "random_state": 42}

    # FIX error "Run ... not found" di GitHub Actions:
    # Kalau MLflow Project sudah bikin run parent, kita bikin nested run.
    if mlflow.active_run() is not None:
        run_ctx = mlflow.start_run(run_name=f"RF_baseline_{params['n_estimators']}_{params['max_depth']}", nested=True)
    else:
        run_ctx = mlflow.start_run(run_name=f"RF_baseline_{params['n_estimators']}_{params['max_depth']}")

    with run_ctx:
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        print(f"[OK] Training selesai. Accuracy test = {acc:.4f}")
        print("[OK] Artefak & metrik tersimpan di MLflow Tracking.")


if __name__ == "__main__":
    main()
