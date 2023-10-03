"""
Test file to define fixtures of Movie Recomendation System
Author: Morsinaldo Medeiros
Date: 2023-09-26
"""
# import libraries
import os
import pytest
import pandas as pd
from dotenv import load_dotenv
from utils import download_data

load_dotenv()

@pytest.fixture
def load_input_data() -> tuple:
    """
    Load the input data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of DataFrames
    """
    # import the data
    if os.path.exists("ml-25m"):
        movies_df = pd.read_csv("ml-25m/movies.csv")
        ratings_df = pd.read_csv("ml-25m/ratings.csv")
    else:
        download_data(os.getenv("URL"), os.getenv("ZIP_FILENAME"))

    return movies_df, ratings_df
