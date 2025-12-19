FROM python:3.9-slim

WORKDIR /app

COPY MLProject /app/MLProject
COPY namadataset_preprocessing /app/namadataset_preprocessing

RUN pip install --no-cache-dir mlflow pandas scikit-learn

CMD ["mlflow", "run", "MLProject", "--env-manager=local", "-P", "data_path=namadataset_preprocessing/data_bersih_eksperimen.csv", "-P", "target_col=Sleep Disorder", "-P", "experiment_name=CI_Retrain_RF_Baseline"]
