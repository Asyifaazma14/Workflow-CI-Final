import os
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_CLEAN_PATH = os.getenv(
    "DATA_CLEAN_PATH",
    "namadataset_preprocessing/data_bersih_eksperimen.csv"
)

# ❗ PENTING: TANPA SPASI
TARGET_COL = os.getenv("TARGET_COL", "Sleep Disorder")

EXPERIMENT_NAME = os.getenv(
    "EXPERIMENT_NAME",
    "CI_Retrain_RF_Baseline"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "file:./mlruns"
)


def main():
    df = pd.read_csv(DATA_CLEAN_PATH)
    df.columns = df.columns.str.strip()

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"TARGET_COL '{TARGET_COL}' tidak ditemukan. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # WAJIB AUTLOG – SESUAI MENTOR
    mlflow.sklearn.autolog(log_models=True)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    with mlflow.start_run(run_name="RF_baseline"):
        model.fit(X_train, y_train)
        model.score(X_test, y_test)

    print("✅ Training selesai")


if __name__ == "__main__":
    main()
