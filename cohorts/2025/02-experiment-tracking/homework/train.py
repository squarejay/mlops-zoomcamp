import os
import pickle
from datetime import datetime

import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data-path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
@click.option("--experiment_name", default="random-forest-homework", help="Name of the experiment")
@click.option(
    "--tracking-uri", default="http://localhost:5555", help="Tracking URI of the MLflow server"
)
def run_train(data_path: str, experiment_name: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    with mlflow.start_run(
        run_name=f"{experiment_name}-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ):
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)


if __name__ == "__main__":
    run_train()
