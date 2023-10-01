"""
Test file of Movie Recomendation System
Author: Morsinaldo Medeiros
Date: 2023-09-26
"""
# import libraries
import pandas as pd

def test_columns_type_of_input_data(load_input_data) -> None:
    """
    Test the columns type of the input data

    Args:
        load_input_data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple of DataFrames

    Returns:
        None
    """
    movies_df, ratings_df = load_input_data
    
    assert movies_df["movieId"].dtype == "int64"
    assert movies_df["title"].dtype == "object"
    assert movies_df["genres"].dtype == "object"
    assert ratings_df["userId"].dtype == "int64"
    assert ratings_df["movieId"].dtype == "int64"
    assert ratings_df["rating"].dtype == "float64"
    assert ratings_df["timestamp"].dtype == "int64"

def test_columns_name_of_input_data(load_input_data) -> None:
    """
    Test the columns name of the input data

    Args:
        load_input_data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple of DataFrames

    Returns:
        None
    """
    movies_df, ratings_df = load_input_data
    
    assert list(movies_df.columns) == ["movieId", "title", "genres"]
    assert list(ratings_df.columns) == ["userId", "movieId", "rating", "timestamp"]

def test_verify_minimum_number_of_movies(load_input_data) -> None:
    """
    Test the minimum number of movies

    Args:
        load_input_data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple of DataFrames

    Returns:
        None
    """
    movies_df, _ = load_input_data
    
    assert movies_df.shape[0] >= 1000

