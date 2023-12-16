import os
import mlflow
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from mlflow import tracking

sns.set_theme(style="whitegrid")

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()


def get_raw_data_artifact():
    """
    Get the raw data artifact.
    """

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Initialize the MlflowClient
    client = tracking.MlflowClient()

    # Hardcoded experiment name
    experiment_name = "Multiclass Text Classification"

    # Get the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise Exception(f"The experiment '{experiment_name}' does not exist.")

    # Search for the latest runs in the experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"])
    
    if runs:

        # search for the run with the name 'fetch_data'
        run = [run for run in runs if run.data.tags.get("mlflow.runName") == "fetch_data"][0]

        # Assuming the first run is the latest
        run_id = run.info.run_id

        data_dir = "./artifacts"
        # Create the 'data' directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Download the artifact
        client.download_artifacts(run_id, "bbc-text.csv", data_dir)
    else:
        raise Exception(f"No runs found for experiment '{experiment_name}'.")

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

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Multiclass Text Classification")

    # get the raw data artifact
    logger.info("Getting the raw data artifact...")
    get_raw_data_artifact()
    logger.info("Raw data artifact downloaded successfully!")

    with mlflow.start_run(run_name="eda"):
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
