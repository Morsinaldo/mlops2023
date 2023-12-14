import ktrain
import mlflow
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

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

def data_segregation(artifact_folder: str):

    with mlflow.start_run():
        
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