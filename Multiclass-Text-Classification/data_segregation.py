import os
import ktrain
import mlflow
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from mlflow import tracking

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def get_clean_data_artifact():
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

        # search for the run with the name 'preprocessing'
        run = [run for run in runs if run.data.tags.get("mlflow.runName") == "preprocessing"][0]

        # Assuming the first run is the latest
        run_id = run.info.run_id

        data_dir = "./artifacts"
        # Create the 'data' directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Download the artifact
        client.download_artifacts(run_id, "bbc-text-preprocessed.csv", data_dir)
    else:
        raise Exception(f"No runs found for experiment '{experiment_name}'.")

def data_segregation(artifact_folder: str):
    """
    Segregate the data into train and test.
    
    Parameters
    ----------
    artifact_folder : str
        Folder to save the data.
    """

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Multiclass Text Classification")

    # get the raw data artifact
    logger.info("Getting the raw data artifact...")
    get_clean_data_artifact()
    logger.info("Raw data artifact downloaded successfully!")

    with mlflow.start_run(run_name="data_segregation"):
        
        logger.info("Reading the data...")
        try:
            df=pd.read_csv(f"{artifact_folder}/bbc-text-preprocessed.csv", engine='python', encoding='UTF-8')
        except Exception as e:
            logger.error(e)

        # split the data
        logger.info("Splitting the data...")
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['category'], test_size=0.20, random_state=42)

        logger.info(f"Shape of the X_train: {X_train.shape}")
        logger.info(f"Shape of the X_test: {X_test.shape}")
        logger.info(f"Shape of the y_train: {y_train.shape}")
        logger.info(f"Shape of the y_test: {y_test.shape}")

        X_train.to_csv(f"{artifact_folder}/X_train.csv")
        y_train.to_csv(f"{artifact_folder}/y_train.csv")
        X_test.to_csv(f"{artifact_folder}/X_test.csv")
        y_test.to_csv(f"{artifact_folder}/y_test.csv")

        # log the artifact
        logger.info("Logging the artifacts...")
        mlflow.log_artifact(f"{artifact_folder}/X_train.csv")
        mlflow.log_artifact(f"{artifact_folder}/y_train.csv")
        mlflow.log_artifact(f"{artifact_folder}/X_test.csv")
        mlflow.log_artifact(f"{artifact_folder}/y_test.csv")