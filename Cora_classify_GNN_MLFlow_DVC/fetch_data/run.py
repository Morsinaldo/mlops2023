import os
import mlflow
import logging
import pandas as pd
from tensorflow import keras

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# mlflow server --host 127.0.0.1 --port 5000

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start MLflow run
with mlflow.start_run():

    zip_file = keras.utils.get_file(
        fname="cora.tgz",
        origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(zip_file), "cora")

    citations = pd.read_csv(
        os.path.join(data_dir, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )
    logger.info("Citations shape: %s", citations.shape)

    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers = pd.read_csv(
        os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
    )

    if not os.path.exists("data"):
        os.makedirs("data")

    # save the data
    logger.info("Saving the data...")
    citations.to_csv("data/citations.csv", index=False)
    papers.to_csv("data/papers.csv", index=False)
    logger.info("Done!")

    # Log artifacts using MLflow
    mlflow.log_artifacts("data")