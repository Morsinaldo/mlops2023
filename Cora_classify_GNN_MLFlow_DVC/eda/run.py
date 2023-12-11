import os
import mlflow
import logging
import pandas as pd
import matplotlib.pyplot as plt

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start MLflow run
with mlflow.start_run():
    # read files
    logger.info("Reading files...")
    citations = pd.read_csv("../fetch_data/data/citations.csv")
    papers = pd.read_csv("../fetch_data/data/papers.csv")

    # log the shape of the data
    logger.info("Citations shape:", citations.shape)
    logger.info("Papers shape:", papers.shape)

    # log the head of the data
    logger.info("Citations head:", citations.head())
    logger.info("Papers head:", papers.head())

    if not os.path.exists("figures"):
        os.makedirs("figures")

    # plot and save a figure with the number of papers per subject
    logger.info("Plotting and saving the figure...")
    papers["subject"].value_counts().plot(kind="bar")
    plt.savefig("figures/subject_counts.png")

    # Log artifacts using MLflow
    mlflow.log_artifacts("figures")

    logger.info("Done!")