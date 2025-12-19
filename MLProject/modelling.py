import argparse
import os
import re

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def clean_text(s: str) -> str:
    """Strip whitespace + hapus quote pembungkus (' atau ") + rapihin whitespace."""
    if s is None:
        return ""
    s = str(s).strip()
    # hapus quote pembungkus berulang (misal: ''Sleep Disorder'' / "Sleep Disorder")
    while (len(s) >= 2) and ((s[0] == s[-1]) and (s[0] in ["'", '"'])):
        s = s[1:-1].strip()
    # rapihin multiple spaces jadi 1
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--experiment_name", type=str, required=True)
    p.add_argument("--tracking_uri", type=str, default="file:./mlruns")
    return p.parse_args()


def main():
    args = parse_args()

    data_path = clean_text(args.data_path)
    target_in = clean_text(args.target_col)
    exp_name = clean_text(args.experiment_name)
    tracking_uri = clean_text(args.tracking_uri)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

    df = pd.read_csv(data_path)

    # normalisasi nama kolom CSV
    df.columns = [clean_text(c) for c in df.columns]

    # cari kolom target secara robust (case-insensitive)
    col_map = {clean_text(c).lower(): c for c in df.columns}
    key = target_in.lower()

    if key not in col_map:
        raise ValueError(
            f"Target col '{target_in}' tidak ada.\n"
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    target_col = col_map[key]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # CI: pakai file store biar gak butuh server 127.0.0.1
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    # WAJIB autolog (tanpa log manual)
    mlflow.sklearn.autolog(log_models=True)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    _ = model.score(X_test, y_test)

    print("Training selesai. Cek mlruns artifact di workflow.")


if __name__ == "__main__":
    main()
