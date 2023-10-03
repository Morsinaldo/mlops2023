"""
Utils functions to perform EDA and Data Cleaning on the Heart Disease dataset
Author: Morsinaldo Medeiros
Date: 2023-10-01
"""
# import libraries
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# set the logging level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def eda_heart_disease(df: pd.DataFrame) -> None:
    """
    Function to perform EDA on the Heart Disease dataset

    Args:
        df (pd.DataFrame): The dataframe to be analyzed

    Returns:
        None
    """
    logging.info("Columns type \n %s", df.dtypes)
    logging.info("Describe of numerical features \n %s", df.describe())

    logging.info("Missing values in dataset: %s", df.isna().sum())

    categorical_cols = ["Sex", "ChestPainType", "FastingBS", "RestingECG",
                        "ExerciseAngina", "ST_Slope", "HeartDisease"]

    fig = plt.figure(figsize=(16, 15))

    for idx, col in enumerate(categorical_cols):
        ax = plt.subplot(4, 2, idx + 1)
        sns.countplot(x=df[col], ax=ax)
        # Adicione rótulos de dados a cada barra
        for container in ax.containers:
            ax.bar_label(container, label_type="center")

    # Salvar o gráfico em um arquivo de imagem (por exemplo, PNG)
    plt.savefig("./images/categorial_features_proportion.png")

    fig = plt.figure(figsize=(16,15))

    fig.suptitle("Proportion of categorical features grouped by HeartDisease", fontsize=16)

    for idx, col in enumerate(categorical_cols[:-1]):
        ax = plt.subplot(4, 2, idx+1)
        # group by HeartDisease
        sns.countplot(x=df[col], hue=df["HeartDisease"], ax=ax)
        # add data labels to each bar
        for container in ax.containers:
            ax.bar_label(container, label_type="center")

    # Salvar o gráfico em um arquivo de imagem (por exemplo, PNG)
    plt.savefig("./images/categorial_features_proportion_grouped.png")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to replace the zero values in the RestingBP 
    and Cholesterol columns with the median of the respective group.

    Args:
        df (pd.DataFrame): The dataframe to be cleaned

    Returns:
        df_clean (pd.DataFrame): The cleaned dataframe
    """

    # only keep non-zero values for RestingBP
    df_clean = df[df["RestingBP"] != 0]
    logging.info("Dataset has %s non-zero values for RestingBP", df_clean.shape[0])

    # get the lines where HeartDisease is 0
    df_heartdisease_mask = df_clean["HeartDisease"]==0
    logging.info("Dataset has %s lines where HeartDisease is 0", df_heartdisease_mask.sum())

    # filter the lines where HeartDisease is 0
    df_cholesterol_without_heartdisease = df_clean.loc[df_heartdisease_mask, "Cholesterol"]
    df_cholesterol_with_heartdisease = df_clean.loc[~df_heartdisease_mask, "Cholesterol"]

    logging.info("Replace the zero values with the median of the respective group")
    # replace the zero values with the median of the respective group
    median_cwh = df_cholesterol_without_heartdisease.median()
    df_clean.loc[df_heartdisease_mask, "Cholesterol"] = \
        df_cholesterol_without_heartdisease.replace(to_replace = 0, value = median_cwh)
    df_clean.loc[~df_heartdisease_mask, "Cholesterol"] = \
        df_cholesterol_with_heartdisease.replace(to_replace = 0, value = median_cwh)

    logging.info("Describe of RestingDB and Cholesterol \n %s", df_clean.describe())

    return df_clean
