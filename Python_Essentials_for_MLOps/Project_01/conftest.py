"""
Test file to define fixtures of Movie Recomendation System
Author: Morsinaldo Medeiros
Date: 2023-09-26
"""
# import libraries
import pytest
import pandas as pd

@pytest.fixture
def load_input_data() -> tuple:
    """
    Load the input data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of DataFrames
    """
    # import the data
    movies_df = pd.read_csv("data/movies.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    
    return movies_df, ratings_df