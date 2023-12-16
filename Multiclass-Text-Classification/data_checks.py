import os
import mlflow
import pytest
import logging
import subprocess

import pandas as pd

from mlflow import tracking
from data_segregation import get_clean_data_artifact

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

@pytest.fixture
def load_data():
    """
    Load the data.
    """
    # download the clean data artifact
    get_clean_data_artifact()

    # load the data
    df = pd.read_csv("./artifacts/bbc-text-preprocessed.csv", engine='python', encoding='UTF-8')

    return df

def test_column_names(load_data):
    """
    Test the column names.
    """
    # columns names: category,text,lower_case,alphabatic,without-link,Special_word,stop_words,short_word,string,Text
    assert load_data.columns.tolist() == ['category','text','lower_case','alphabatic','without-link','Special_word','stop_words','short_word','string','Text']

def test_column_dtypes(load_data):
    """
    Test the column dtypes.
    """
    assert load_data.dtypes.tolist() == [object, object, object, object, object, object, object, object, object, object]

def test_column_values(load_data):
    """
    Test the column values.
    """

    categories = load_data['category'].unique().tolist()

    assert categories == ['tech', 'business', 'sport', 'entertainment', 'politics']

def test_minimum_rows(load_data):
    """
    Test the minimum number of rows.
    """
    assert load_data.shape[0] >= 1000

def run_tests():
    """
    Run tests.
    """
    
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Hardcoded experiment name
    experiment_name = "Multiclass Text Classification"

    with mlflow.start_run(run_name="data_checks"):
        logger.info("Starting Data Checks...")

        # run tests with pytest
        test_output = subprocess.run(['pytest', '-v', os.path.abspath(__file__)], capture_output=True, text=True)

        logger.info("Test Output:\n" + test_output.stdout)

        if test_output.stderr:
            logger.error("Test Errors:\n" + test_output.stderr)