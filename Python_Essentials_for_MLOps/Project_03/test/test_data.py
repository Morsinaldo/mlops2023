"""
Test module for the Heart Disease dataset
Author: Morsinaldo Medeiros
Date: 2023-10-01
"""
# import libraries
import pandas as pd

def test_dataset_loaded_correctly(df_heart_disease: pd.DataFrame) -> None:
    """
    Test to check if the dataset was loaded correctly

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    assert not df_heart_disease.empty

def test_columns_present(df_heart_disease)  -> None:
    """
    Test to check if the dataset has the expected columns

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    expected_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                        "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", 
                        "Oldpeak", "ST_Slope", "HeartDisease"]
    assert all(col in df_heart_disease.columns for col in expected_columns)

def test_column_data_types(df_heart_disease) -> None:
    """
    Test to check if the data types of columns in the dataset match the expected types.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    expected_data_types = {
        "Age": "int64",
        "Sex": "object",
        "ChestPainType": "object",
        "RestingBP": "int64",
        "Cholesterol": "int64",
        "FastingBS": "int64",
        "RestingECG": "object",
        "MaxHR": "int64",
        "ExerciseAngina": "object",
        "Oldpeak": "float64",
        "ST_Slope": "object",
        "HeartDisease": "int64"
    }

    for column, expected_type in expected_data_types.items():
        assert str(df_heart_disease[column].dtype) == expected_type

def test_sex_categories(df_heart_disease) -> None:
    """
    Test to check if the 'Sex' column contains valid categories.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    expected_categories = ["M", "F"]
    assert all(category in df_heart_disease["Sex"].unique() for category in expected_categories)

def test_chest_pain_type_categories(df_heart_disease) -> None:
    """
    Test to check if the 'ChestPainType' column contains valid categories.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    expected_categories = ["TA", "ATA", "NAP", "ASY"]
    assert all(category in df_heart_disease["ChestPainType"].unique() \
               for category in expected_categories)

def test_fasting_bs_categories(df_heart_disease) -> None:
    """
    Test to check if the 'FastingBS' column contains valid categories.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    expected_categories = [0, 1]
    assert all(category in df_heart_disease["FastingBS"].unique() \
               for category in expected_categories)

def test_resting_ecg_categories(df_heart_disease) -> None:
    """
    Test to check if the 'RestingECG' column contains valid categories.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    expected_categories = ["Normal", "ST", "LVH"]
    assert all(category in df_heart_disease["RestingECG"].unique() \
               for category in expected_categories)

def test_exercise_angina_categories(df_heart_disease) -> None:
    """
    Test to check if the 'ExerciseAngina' column contains valid categories.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    expected_categories = ["Y", "N"]
    assert all(category in df_heart_disease["ExerciseAngina"].unique() \
               for category in expected_categories)

def test_heart_disease_categories(df_heart_disease) -> None:
    """
    Test to check if the 'HeartDisease' column contains valid categories.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    expected_categories = [0, 1]
    assert all(category in df_heart_disease["HeartDisease"].unique() \
               for category in expected_categories)

def test_age_range(df_heart_disease) -> None:
    """
    Test to check the range of values in the 'Age' column.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    assert df_heart_disease["Age"].min() >= 0
    assert df_heart_disease["Age"].max() <= 120

def test_resting_bp_range(df_heart_disease) -> None:
    """
    Test to check the range of values in the 'RestingBP' column.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    assert df_heart_disease["RestingBP"].min() >= 0
    assert df_heart_disease["RestingBP"].max() <= 220

def test_cholesterol_range(df_heart_disease) -> None:
    """
    Test to check the range of values in the 'Cholesterol' column.

    Args:
        df_heart_disease (pd.DataFrame): The Heart Disease dataset

    Returns:
        None
    """
    assert df_heart_disease["Cholesterol"].min() >= 0
    assert df_heart_disease["Cholesterol"].max() <= 650
