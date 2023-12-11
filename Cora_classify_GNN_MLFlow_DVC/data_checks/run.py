import os
import pytest
import mlflow
import logging
import pandas as pd
from tensorflow import keras

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# mlflow server --host 127.0.0.1 --port 5000

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

@pytest.fixture
def data():
    papers = pd.read_csv("data_cleaned/papers.csv")
    citations = pd.read_csv("data_cleaned/citations.csv")
    return papers, citations

# Start MLflow run
with mlflow.start_run():

    def test_data(data):
        papers, citations = data
        
        # verify if all the columns are present
        assert "paper_id" in papers.columns
        assert "subject" in papers.columns
        assert "source" in citations.columns
        assert "target" in citations.columns

    def test_is_not_empty(data):
        papers, citations = data
        
        # verify if the data is not empty
        assert not papers.empty
        assert not citations.empty

    def test_is_not_null(data):
        papers, citations = data
        
        # verify if the data is not null
        assert not papers.isnull().values.any()
        assert not citations.isnull().values.any()