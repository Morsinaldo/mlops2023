"""
Test the podcast DAG.
Author: Morsinaldo Medeiros
Date: 2023-09-30
"""
import logging
from vosk import Model
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from dags.podcast import create_database, fetch_data, initialize_transcription_model

# set the logging level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def test_create_database() -> None:
    """
    Test that the create_database task is in the DAG 
    and has the correct upstream dependencies.
    """

    database = create_database()

    assert isinstance(database, SQLExecuteQueryOperator)

def test_fetch_data() -> None:
    """
    Test that the fetch_data function returns a list of episodes.
    """

    episodes = fetch_data()

    assert isinstance(episodes, list)
    assert len(episodes) > 0

def test_load_transcription_model() -> None:
    """
    Test that the transcription model is loaded.
    """

    model = initialize_transcription_model()

    assert isinstance(model, Model)
