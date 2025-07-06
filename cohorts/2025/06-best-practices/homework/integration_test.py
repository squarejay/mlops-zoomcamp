#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
from datetime import datetime
from batch import get_input_path, get_output_path


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def create_test_data():
    # Same test data from Q3
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
    
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)
    
    return df_input


def save_test_data():
    # Create test data
    df_input = create_test_data()
    
    # This is data for January 2023
    year = 2023
    month = 1
    
    # Get the input file path using our environment variables
    input_file = get_input_path(year, month)
    
    # Set up S3 options for LocalStack
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    
    print(f"Saving test data to: {input_file}")
    
    # Save using the exact snippet from homework
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )
    
    print("Test data saved successfully!")
    return year, month


def run_batch_job(year, month):
    # Run the batch.py script using os.system
    print(f"Running batch job for {year:04d}-{month:02d}")
    command = f"python batch.py {year} {month}"
    result = os.system(command)
    
    if result == 0:
        print("Batch job completed successfully!")
    else:
        print(f"Batch job failed with exit code: {result}")
    
    return result == 0


def read_and_verify_results(year, month):
    # Get the output file path
    output_file = get_output_path(year, month)
    
    # Set up S3 options for LocalStack
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    
    print(f"Reading results from: {output_file}")
    
    # Read the result data
    df_result = pd.read_parquet(output_file, storage_options=options)
    
    print("Result DataFrame:")
    print(df_result)
    
    # Calculate the sum of predicted durations
    sum_predicted_durations = df_result['predicted_duration'].sum()
    print(f"Sum of predicted durations: {sum_predicted_durations}")
    
    return sum_predicted_durations


def main():
    # Step 1: Save test data to S3
    year, month = save_test_data()
    
    # Step 2: Run the batch job
    success = run_batch_job(year, month)
    
    if success:
        # Step 3: Read and verify results
        sum_durations = read_and_verify_results(year, month)
        print(f"Sum of predicted durations: {sum_durations}")
    else:
        print("Integration test failed!")


if __name__ == '__main__':
    main() 