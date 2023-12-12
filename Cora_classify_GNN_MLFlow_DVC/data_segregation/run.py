import os
import mlflow
import logging
import argparse
import numpy as np
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
    citations = pd.read_csv(f"../{args.artifact_folder}/citations_cleaned.csv")
    papers = pd.read_csv(f"../{args.artifact_folder}/papers_cleaned.csv")

    # split the data into train and test
    logger.info("Splitting the data into train and test...")
    train_data, test_data = [], []

    for _, group_data in papers.groupby("subject"):
        # Select around 50% of the dataset for training.
        random_selection = np.random.rand(len(group_data.index)) <= 0.5
        train_data.append(group_data[random_selection])
        test_data.append(group_data[~random_selection])

    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)

    logger.info("Train data shape:", train_data.shape)
    logger.info("Test data shape:", test_data.shape)

    if not os.path.exists(f"../{args.artifact_folder}"):
        os.makedirs(f"../{args.artifact_folder}")

    # save the data
    logger.info("Saving the data...")
    train_data.to_csv(f"../{args.artifact_folder}/train_data.csv", index=False)
    test_data.to_csv(f"../{args.artifact_folder}/test_data.csv", index=False)

    # Log artifacts using MLflow
    # mlflow.log_artifacts("data")
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cora classification with GNNs")

    parser.add_argument(
        "--artifact_folder",
        type=str,
        default="artifacts",
        help="Folder where the data will be saved",
    )

    args = parser.parse_args()

    process_args(args)

# Run this script with:
    # mlflow run . -P artifact_folder=artifacts