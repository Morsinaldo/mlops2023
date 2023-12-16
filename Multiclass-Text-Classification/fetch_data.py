import os
import mlflow
import logging
import requests

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# mlflow server --host 127.0.0.1 --port 5000

def fetch_data(url: str, artifact_folder: str):
    """
    Download the data from the url and save it in the artifact folder.

    Parameters
    ----------
    url : str
        URL of the data.
    artifact_folder : str
        Folder to save the data.
    """
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Multiclass Text Classification")

    with mlflow.start_run(run_name="fetch_data"):
 
        if not os.path.exists(artifact_folder):
            os.makedirs(artifact_folder)

        # Download the data
        try:
            requests.get(url)

            # Download the data
            logger.info("Downloading the data...")
            data = requests.get(url).content

            # Save the data
            logger.info("Saving the data...")
            with open(f"{artifact_folder}/bbc-text.csv", "wb") as f:
                f.write(data)
        except Exception as e:
            logger.error(e)

        # log the artifact
        mlflow.log_artifact(f"{artifact_folder}/bbc-text.csv")
        logger.info("Data downloaded successfully!")

    
