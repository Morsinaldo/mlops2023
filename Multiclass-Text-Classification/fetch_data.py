import os
import mlflow
import logging

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()


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
    with mlflow.start_run():
        if not os.path.exists(artifact_folder):
            os.makedirs(artifact_folder)

        # Download the data
        try:
            os.system(f"wget {url} -O {artifact_folder}/bbc-text.csv")
        except Exception as e:
            logger.error(e)

        # log the artifact
        mlflow.log_artifact(f"{artifact_folder}/bbc-text.csv")
        logger.info("Data downloaded successfully!")
    
