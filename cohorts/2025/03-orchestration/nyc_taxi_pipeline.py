import logging
from pathlib import Path

import mlflow
import pandas as pd
import requests
from prefect import flow, get_run_logger, task
from prefect.task_runners import ConcurrentTaskRunner
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

ROOT_DIR = Path(__file__).parent / "data"


@task(
    name="download_nyc_taxi_data",
    description="Download NYC taxi dataset",
    retries=3,
    retry_delay_seconds=60,
)
def download_data(month: str, logger: logging.Logger) -> pd.DataFrame:
    """Download NYC taxi data from the official TLC website."""
    file_name = f"yellow_tripdata_2023-{month}.parquet"
    file_path = ROOT_DIR / file_name

    if not file_path.exists():
        # Using a sample dataset URL (2023 January Yellow Taxi data)
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{file_name}"

        # Download the parquet file
        response = requests.get(url, timeout=300)
        response.raise_for_status()

        # Save to local file
        with open(file_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Download from {url} complete")

    else:
        logger.info(f"File {file_name} already exists. Skipping download")

    # Read the parquet file
    df = pd.read_parquet(file_path)

    logger.info(f"The dataset contains {len(df)} rows and {len(df.columns)} columns")

    return df


@task(name="clean_data")
def clean_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    logger.info(f"Cleaned the data. It now contains {len(df)} rows")

    return df


@task(name="train_model")
def train_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    logger: logging.Logger,
) -> LinearRegression:
    mlflow.set_tracking_uri("http://127.0.0.1:5555")
    mlflow.set_experiment("mlops-zoomcamp-homework-3")
    mlflow.sklearn.autolog()

    categorical = ["PULocationID", "DOLocationID"]
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values

    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_train = lr.predict(X_train)
        y_pred_val = lr.predict(X_val)

        metrics = {
            "train_rmse": root_mean_squared_error(y_train, y_pred_train),
            "val_rmse": root_mean_squared_error(y_val, y_pred_val),
            "intercept": lr.intercept_,
        }

        mlflow.log_metrics(metrics)

        logger.info(metrics)

    return lr


@flow(
    name="NYC taxi analysis",
    task_runner=ConcurrentTaskRunner(),
)
def nyc_taxi_pipeline():
    logger = get_run_logger()

    months = {
        "02": None,
        "03": None,
    }

    for month in months.keys():
        downloaded_data = download_data(month, logger)
        cleaned_data = clean_data(downloaded_data, logger)

        months[month] = cleaned_data

    train_model(
        df_train=months["03"],
        df_val=months["02"],
        logger=logger,
    )

    return None


if __name__ == "__main__":
    nyc_taxi_pipeline()
