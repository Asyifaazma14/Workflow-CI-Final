import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def _clean_text(x: str) -> str:
    # buang spasi dan kutip yang sering kebawa dari YAML/CLI
    if x is None:
        return x
    x = str(x).strip()
    # hapus kutip pembuka/penutup kalau ada
    if (x.startswith('"') and x.endswith('"')) or (x.startswith("'") and x.endswith("'")):
        x = x[1:-1].strip()
    # kalau ada kutip nyangkut di tengah (contoh: "'Sleep Disorder '")
    x = x.strip("'").strip('"').strip()
    return x


def parse_args():
    p = argparse.ArgumentParser()

    # Terima BOTH: --data_path dan --data-path (biar aman)
    p.add_argument("--data_path", "--data-path", dest="data_path", type=str, required=True)
    p.add_argument("--target_col", "--target-col", dest="target_col", type=str, required=True)
    p.add_argument("--experiment_name", "--experiment-name", dest="experiment_name", type=str, required=True)
    p.add_argument("--tracking_uri", "--tracking-uri", dest="tracking_uri", type=str, required=True)

    return p.parse_args()


def main():
    args = parse_args()

    args.data_path = _clean_text(args.data_path)
    args.target_col = _clean_text(args.target_col)
    args.experiment_name = _clean_text(args.experiment_name)
    args.tracking_uri = _clean_text(args.tracking_uri)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df.columns = df.columns.astype(str).str.strip()

    # kalau user salah casing/spasi, coba cari kolom paling cocok
    if args.target_col not in df.columns:
        # fallback: case-insensitive match
        lowered = {c.lower(): c for c in df.columns}
        key = args.target_col.lower().strip()
        if key in lowered:
            args.target_col = lowered[key]
        else:
            raise ValueError(
                f"Target col '{args.target_col}' tidak ada. Kolom tersedia: {df.columns.tolist()}"
            )

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    
    mlflow.sklearn.autolog(log_models=True)

    params = {"n_estimators": 200, "max_depth": 10}

    with mlflow.start_run(run_name=f"RF_baseline_{params['n_estimators']}_{params['max_depth']}"):
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)
        _ = model.score(X_test, y_test)

    print("[OK] Training selesai. Artefak & metrik tersimpan di MLflow.")


if __name__ == "__main__":
    main()
