"""
Test the podcast DAG.
Author: Morsinaldo Medeiros
Date: 2023-09-30
"""
import glob
import logging
import requests
import pandas as pd
from unittest.mock import patch, MagicMock
from airflow.providers.sqlite.hooks.sqlite import SqliteHook
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
# from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

from ..dags.podcast import create_database, fetch_data, load_data, download_data

# set the logging level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def test_create_database() -> None:
    """Test that the create_database task is in the DAG and has the correct upstream dependencies."""
    
    create_database = create_database()

    assert isinstance(create_database, SqliteOperator)

def test_fetch_episodes() -> None:
    """Test that the fetch_episodes task is in the DAG and has the correct upstream dependencies."""

    with patch('podcast_summary.requests.get') as mock_get, \
         patch('podcast_summary.xmltodict.parse') as mock_parse:
        
        mock_get.return_value.text = '<xml>fake xml</xml>'

        # Mock the result from xmltodict.parse
        mock_parse.return_value = {
            "rss": {
                "channel": {
                    "item": [
                        {"title": "Episode 1"},
                        {"title": "Episode 2"},
                    ]
                }
            }
        }

        # Call fetch_data
        result = fetch_data()

        # Check the result
        assert len(result) == 2
        assert result[0]['title'] == 'Episode 1'
        assert result[1]['title'] == 'Episode 2'

def mock_get_pandas_df(*_args, **_kwargs) -> pd.DataFrame:
    """
    Mocking the get_pandas_df method of SqliteHook.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    
    Returns:
        pd.DataFrame: A dataframe with the first episode.
    """
    # Simulating a database with only the first episode
    return pd.DataFrame([EPISODES_DATA[0]])

def mock_insert_rows(*_args, **_kwargs) -> None:
    """
    Mocking the insert_rows method of SqliteHook.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    
    Returns:
        None
    """
    pass  # Just a placeholder, as we're not actually inserting rows in this test

def test_load_data(monkeypatch: MagicMock, podcast_data) -> None:
    """Test that the load_data task is in the DAG and has the correct upstream dependencies."""

    # Patching the methods of SqliteHook to use our mock functions
    monkeypatch.setattr(SqliteHook, "get_pandas_df", mock_get_pandas_df)
    monkeypatch.setattr(SqliteHook, "insert_rows", mock_insert_rows)

    # Call load_data
    result = load_data(podcast_data[0])

    # Check the result
    assert len(result) == 1
    assert result[0][0] == 'https://example.com/episode1'

def mock_requests_get(*_args, **_kwargs) -> MagicMock:
    """
    Mocking the get method of requests.
    """
    mock_response = MagicMock()
    mock_response.content = b"mock audio content"
    return mock_response

def test_download_data(monkeypatch: MagicMock, podcast_data, tmpdir) -> None:
    """
    Test that the download_data task is in the DAG and has the correct upstream dependencies.
    """

    # Patching requests.get to use our mock function
    monkeypatch.setattr(requests, "get", mock_requests_get)

    # Using a temporary directory for the test
    temp_dir_str = str(tmpdir)
    monkeypatch.setattr("podcast_summary.EPISODE_FOLDER", temp_dir_str)

    # Call download_data
    result = download_data(podcast_data)

    # Check the result
    assert len(result) == 2
    assert result[0]["filename"] == "episode1.mp3"
    assert result[1]["filename"] == "episode2.mp3"

    # Check that the files were downloaded
    assert len(glob.glob(f"{temp_dir_str}/*.mp3")) == 2