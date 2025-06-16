#!/usr/bin/env python

import pickle
import pandas as pd
import click
import logging
from typing import Optional

CATEGORICAL_FEATURES = ['PULocationID', 'DOLocationID']

def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna(-1).astype('int').astype('str')

    return df

@click.command()
@click.option("--year", type=int, default=2023)
@click.option("--month", type=int, default=3)
@click.option("--model-path", type=str, default="model.bin")
@click.option("--output-path", type=str, default=None)
def main(year: int, month: int, model_path: str, output_path: Optional[str] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    data_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    logger.info(f"Reading data from {data_url}")
    df = read_data(data_url)

    logger.info(f"Number of rows: {len(df)}")

    dicts = df[CATEGORICAL_FEATURES].to_dict(orient='records')
    X_val = dv.transform(dicts)
    
    logger.info("Running predictions")
    y_pred = model.predict(X_val)

    logger.info(f"Mean predicted duration: {y_pred.mean():.2f}")
    logger.info(f"Standard deviation of predicted duration: {y_pred.std():.2f}")

    if output_path is not None:
        logger.info(f"Saving results to {output_path}")
        df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
        df_result = df[["ride_id"]]
        df_result["y_pred"] = y_pred

        df_result.to_parquet(
            output_path,
            engine='pyarrow',
            compression=None,
            index=False
        )

if __name__ == "__main__":
    main()