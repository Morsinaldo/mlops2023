import os
import mlflow
import logging
import argparse
import pandas as pd
import networkx as nx
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

    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

    papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
    citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
    citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
    papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

    plt.figure(figsize=(10, 10))
    colors = papers["subject"].tolist()
    cora_graph = nx.from_pandas_edgelist(citations.sample(n=1500))
    subjects = list(papers[papers["paper_id"].isin(list(cora_graph.nodes))]["subject"])
    nx.draw_spring(cora_graph, node_size=15, node_color=subjects)

    if not os.path.exists(f"../{args.figures_folder}"):
        os.makedirs(f"../{args.figures_folder}")

    # plot and save a figure with the number of papers per subject
    logger.info("Plotting and saving the figure...")
    plt.savefig(f"../{args.figures_folder}/cora_graph.png")

    if not os.path.exists(f"../{args.artifact_folder}"):
        os.makedirs(f"../{args.artifact_folder}")

    # save the processed dataset
    logger.info("Saving the data...")
    citations.to_csv(f"../{args.artifact_folder}/citations_cleaned.csv", index=False)
    papers.to_csv(f"../{args.artifact_folder}/papers_cleaned.csv", index=False)

    # Log artifacts using MLflow
    # mlflow.log_artifacts("figures")
    # mlflow.log_artifacts("data_cleaned")
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