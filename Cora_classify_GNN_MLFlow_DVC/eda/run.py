import os
import mlflow
import logging
import argparse
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

def process_args(args):
    # read files
    logger.info("Reading files...")
    citations = pd.read_csv(f"../{args.artifact_folder}/citations.csv")
    papers = pd.read_csv(f"../{args.artifact_folder}/papers.csv")

    # log the shape of the data
    logger.info("Citations shape:", citations.shape)
    logger.info("Papers shape:", papers.shape)

    # log the head of the data
    logger.info("Citations head:", citations.head())
    logger.info("Papers head:", papers.head())

    if not os.path.exists(f"../{args.figures_folder}"):
        os.makedirs(f"../{args.figures_folder}")

    # plot and save a figure with the number of papers per subject
    logger.info("Plotting and saving the figure...")
    papers["subject"].value_counts().plot(kind="bar")
    plt.savefig(f"../{args.figures_folder}/subject_counts.png")

    # Log artifacts using MLflow
    # mlflow.log_artifacts(f"../{args.figures_folder}")

    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cora classification with GNNs")

    parser.add_argument(
        "--figures_folder",
        type=str,
        default="figures",
        help="Folder where the data will be saved",
    )

    parser.add_argument(
        "--artifact_folder",
        type=str,
        default="artifacts",
        help="Folder where the data will be saved",
    )

    args = parser.parse_args()

    process_args(args)

# Run this script with:
    # mlflow run . -P figures_folder=figures artifact_folder=artifacts