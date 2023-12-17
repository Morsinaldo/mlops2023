"""
Python file to train the model.
"""

import os
import logging
import ktrain
import mlflow

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ktrain import text
from mlflow import tracking
from requests.exceptions import RequestException

sns.set_theme(style="whitegrid")

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt='%d-%m-%Y %H:%M:%S'
)

# reference for a logging obj
logger = logging.getLogger()

def get_segregated_data_artifact():
    """
    Get the segregated data artifact.
    """

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Initialize the MlflowClient
    client = tracking.MlflowClient()

    # Hardcoded experiment name
    experiment_name = "Multiclass Text Classification"

    # Get the experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RequestException(f"The experiment '{experiment_name}' does not exist.")

    # Search for the latest runs in the experiment
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"]
    )

    if runs:

        # search for the run with the name 'data_segregation'
        run = [run for run in runs if run.data.tags.get("mlflow.runName") == "data_segregation"][0]

        # Assuming the first run is the latest
        run_id = run.info.run_id

        data_dir = "./artifacts"
        # Create the 'data' directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Download the artifact
        client.download_artifacts(run_id, "X_train.csv", data_dir)
        client.download_artifacts(run_id, "y_train.csv", data_dir)
        client.download_artifacts(run_id, "X_test.csv", data_dir)
        client.download_artifacts(run_id, "y_test.csv", data_dir)
    else:
        raise RequestException(f"No runs found for experiment '{experiment_name}'.")


def train(artifact_folder: str, figures_folder: str):
    """
    Train the model.
    
    Parameters
    ----------
    artifact_folder : str
        Folder to save the data.
    figures_folder : str
        Folder to save the figures.
    """

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Multiclass Text Classification")

    # get the raw data artifact
    logger.info("Getting the segregated data artifact...")
    get_segregated_data_artifact()
    logger.info("Segregated data artifact downloaded successfully!")

    with mlflow.start_run(run_name="train"):

        logger.info("Reading the data...")
        X_train = pd.read_csv(f"{artifact_folder}/X_train.csv")
        y_train = pd.read_csv(f"{artifact_folder}/y_train.csv")
        X_test = pd.read_csv(f"{artifact_folder}/X_test.csv")
        y_test = pd.read_csv(f"{artifact_folder}/y_test.csv")

        # transform to list
        X_train = X_train.text.tolist()
        X_test = X_test.text.tolist()
        y_train = y_train.category.tolist()
        y_test = y_test.category.tolist()

        class_names = ['sport', 'business', 'politics','tech', 'entertainment']

        # load data
        logger.info("Loading the data...")
        (x_train, y_train), (x_val, y_val), preproc = text.texts_from_array(
            x_train=X_train,
            y_train=y_train,
            x_test=X_test,
            y_test=y_test,
            class_names=class_names,
            preprocess_mode='bert',
            maxlen=512,
            max_features=20000
        )

        # load model
        logger.info("Loading the model...")
        model = text.text_classifier('bert', train_data=(x_train,y_train), preproc=preproc)

        # wrap model and data in ktrain.Learner object
        logger.info("Wrapping the model...")
        learner = ktrain.get_learner(
            model,
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            batch_size=6
        )

        # fit the model
        logger.info("Fitting the model...")
        learner.fit_onecycle(2e-5, 3)

        # evaluate the model
        logger.info("Evaluating the model...")
        logger.info(learner.validate(val_data=(x_val,y_val), class_names=class_names))

        # plot accuracy and loss graph
        logger.info("Plotting the accuracy and loss graph...")
        learner.plot()
        plt.savefig(f"./{figures_folder}/acc_loss_graph.png")
        plt.close()

        # plot the confusion matrix
        logger.info("Plotting the confusion matrix...")

        confusion_array = learner.validate(val_data=(x_val, y_val), class_names=class_names)

        # create the figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")

        # save the figure
        plt.savefig(f"./{figures_folder}/confusion_matrix.png")

        # save the model
        predictor = ktrain.get_predictor(learner.model, preproc)
        predictor.save(f"./{artifact_folder}/bert_trained")

        # log the artifact
        logger.info("Logging the artifacts...")
        mlflow.log_artifact(f"./{artifact_folder}/bert_trained")

        # log the confusion matrix
        mlflow.log_artifact(f"./{figures_folder}/confusion_matrix.png")
        mlflow.log_artifact(f"./{figures_folder}/acc_loss_graph.png")

        mlflow.log_metric("val_loss", learner.history.history['val_loss'][0])
