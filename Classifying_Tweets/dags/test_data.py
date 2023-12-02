import pytest
import wandb
import math
import pandas as pd

# This is global so all tests are collected under the same run
run = wandb.init(project="tweet_classifying", job_type="data_checks")

@pytest.fixture(scope="session")
def data():
    # Assume 'run' is a global object in your testing environment
    local_path = run.use_artifact("preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path)
    return df

def test_target_labels(data):
    # Ensure that the 'target' column has only 0 and 1 as labels, excluding NaN
    actual_labels = set(data['target'].unique())

    # Check for equality excluding NaN
    assert all(math.isnan(label) or label in {0.0, 1.0} for label in actual_labels)

def test_dataset_size(data):
    # Ensure that the dataset has at least 1000 rows
    assert len(data) >= 1000

def test_final_column_type(data):
    # Ensure that the 'final' column is of type string
    assert data['final'].dtype == 'O'