"""
Conftest file for the podcast DAG.
Author: Morsinaldo Medeiros
Date: 2023-09-30
"""
import pytest
from airflow.models import DAG
from dags.podcast import podcast_summary
from dags.podcast import fetch_data

# instantiate podcast_summary as a fixture
@pytest.fixture
def airflow_dag() -> DAG:
    """
    Fixture to instantiate the DAG.

    Returns:
        DAG: An instance of the DAG.
    """
    return podcast_summary()

# fixture to simulate fake data
@pytest.fixture
def podcast_episodes() -> list:
    """
    Simulate a list of podcast episodes.

    Returns:
        list: A list of podcast episodes.
    """
    return fetch_data()
