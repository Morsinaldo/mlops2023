import os
import mlflow
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from gnn_node_classifier import GNNNodeClassifier, run_experiment, display_learning_curves

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

# hyperparameters
hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

# Start MLflow run
with mlflow.start_run():

    # read files
    logger.info("Reading files...")
    train_data = pd.read_csv("../data_segregation/data/train_data.csv")
    test_data = pd.read_csv("../data_segregation/data/test_data.csv")
    papers = pd.read_csv("../preprocessing/data_cleaned/papers.csv")
    citations = pd.read_csv("../preprocessing/data_cleaned/citations.csv")

    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

    feature_names = list(set(papers.columns) - {"paper_id", "subject"})
    num_features = len(feature_names)
    num_classes = len(class_idx)

    # Create train and test features as a numpy array.
    x_train = train_data[feature_names].to_numpy()
    x_test = test_data[feature_names].to_numpy()
    # Create train and test targets as a numpy array.
    y_train = train_data["subject"]
    y_test = test_data["subject"]

    # Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
    edges = citations[["source", "target"]].to_numpy().T
    # Create an edge weights array of ones.
    edge_weights = tf.ones(shape=edges.shape[1])
    # Create a node features array of shape [num_nodes, num_features].
    node_features = tf.cast(
        papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32
    )
    # Create graph info tuple with node_features, edges, and edge_weights.
    graph_info = (node_features, edges, edge_weights)

    logger.info("Edges shape:", edges.shape)
    logger.info("Nodes shape:", node_features.shape)


    gnn_model = GNNNodeClassifier(
        graph_info=graph_info,
        num_classes=num_classes,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        name="gnn_model",
    )

    logger.info("GNN output shape:", gnn_model([1, 10, 100]))

    logger.info(gnn_model.summary())

    x_train = train_data.paper_id.to_numpy()
    history = run_experiment(gnn_model, x_train, y_train)

    display_learning_curves(history)

    x_test = test_data.paper_id.to_numpy()
    _, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
    logger.info(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

    # Log metrics using MLflow
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log model using MLflow
    mlflow.tensorflow.log_model(gnn_model, "model")

    logger.info("Done!")