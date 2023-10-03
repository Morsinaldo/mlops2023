"""
This file contains the fixtures used in the tests of the Heart Disease Prediction project.
Author: Morsinaldo Medeiros
Date: 2023-10-01
"""
# import libraries
import pytest
import pandas as pd

@pytest.fixture
def df_heart_disease() -> pd.DataFrame:
    """
    Fixture to load the Heart Disease dataset

    Returns:
        pd.DataFrame: The Heart Disease dataset
    """
    df = pd.read_csv("data/heart_disease_prediction.csv")
    return df
