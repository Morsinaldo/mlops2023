import os
import mlflow
import logging
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

# Start MLflow run
with mlflow.start_run():

    # read files
    logger.info("Reading files...")
    citations = pd.read_csv("../fetch_data/data/citations.csv")
    papers = pd.read_csv("../fetch_data/data/papers.csv")

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

    if not os.path.exists("figures"):
        os.makedirs("figures")

    # plot and save a figure with the number of papers per subject
    logger.info("Plotting and saving the figure...")
    plt.savefig("figures/cora_graph.png")

    if not os.path.exists("data_cleaned"):
        os.makedirs("data_cleaned")

    # save the processed dataset
    logger.info("Saving the data...")
    citations.to_csv("data_cleaned/citations.csv", index=False)
    papers.to_csv("data_cleaned/papers.csv", index=False)

    # Log artifacts using MLflow
    mlflow.log_artifacts("figures")
    mlflow.log_artifacts("data_cleaned")
    logger.info("Done!")