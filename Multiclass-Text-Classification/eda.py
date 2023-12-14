import os
import mlflow
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("Multiclass Text Classification")

def eda(figures_folder: str, artifact_folder: str):
    """
    Exploratory Data Analysis (EDA).

    Parameters
    ----------
    figures_folder : str
        Folder to save the figures.
    artifact_folder : str
        Folder to save the data.
    """
    with mlflow.start_run():
        logger.info("Starting EDA...")

        # read raw data artifact
        try:
            df=pd.read_csv(f"{artifact_folder}/bbc-text.csv", engine='python', encoding='UTF-8')
        except Exception as e:
            logger.error(e)

        logger.info(df.head())

        logger.info(f"Shape of the data: {df.shape}")

        if not os.path.exists(figures_folder):
            os.makedirs(figures_folder)

        # plot the label distribution
        logger.info("Plotting the label distribution...")
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 5))
        sns.countplot(x="category", data=df)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{figures_folder}/category.png")

        # log the artifact
        logger.info("Logging the artifacts...")
        mlflow.log_artifact(f"{figures_folder}/category.png")


