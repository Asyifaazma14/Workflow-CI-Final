FROM python:3.9-slim

WORKDIR /app

# copy SEMUA isi MLProject (karena context sudah MLProject)
COPY . /app

RUN pip install --no-cache-dir mlflow pandas scikit-learn

CMD ["python", "modelling.py"]
