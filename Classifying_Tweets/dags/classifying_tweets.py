import os
import requests
import logging
import wandb
import nltk
import pytest

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from airflow import DAG
from airflow.utils.task_group import TaskGroup
from airflow.operators.python_operator import PythonOperator
from airflow.configuration import conf

from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from codecarbon import EmissionsTracker
from datetime import datetime

from preprocessing_helper import *

logging.basicConfig(
    filename="classifying_tweets.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p"
)


def download_data(url, file_name):
    logging.info(f"Downloading {url} to {file_name}")

    r = requests.get(url)

    with open(file_name, "wb") as f:
        f.write(r.content)

def delete_data(file_name):
    logging.info(f"Deleting {file_name}")

    os.remove(file_name)

def create_subset(file_name):
    logging.info(f"Reading {file_name}")

    data = pd.read_csv(file_name)

    logging.info(f"Data {file_name} shape: {data.shape}")

    data["subset"] = file_name.split('.')[0]

    data.to_csv(f"{file_name.split('.')[0]}_with_subset.csv", index=False)

def concatenate_data():
    logging.info("Concatenating data")

    train = pd.read_csv("train_with_subset.csv")
    test = pd.read_csv("test_with_subset.csv")

    data = pd.concat([train, test], axis=0)

    logging.info(f"Concatenated data shape: {data.shape}")

    data.to_csv("data_concatenated.csv", index=False)

def wandb_login(api_key):
    logging.info("Logging in to Weights & Biases")

    wandb.login(key=api_key)

def wandb_artifact_upload():
    wandb.init(project="tweet_classifying", job_type="upload")

    artifact = wandb.Artifact("raw_data", type="RawData", description="Real and Fake Disaster-Related Tweets Dataset")

    artifact.add_file("data_concatenated.csv", name="raw_data.csv")

    wandb.run.log_artifact(artifact)

    wandb.finish()

def wandb_artifact_download():
    wandb.init(project="tweet_classifying", save_code=True)

    artifact = wandb.use_artifact("raw_data:latest")
    
    artifact_dir = artifact.download()

    logging.info(f"Artifact downloaded to {artifact_dir}")

    wandb.finish()

def create_value_counts_plot(column_name):
    wandb.init(project="tweet_classifying", save_code=True)

    artifact = wandb.use_artifact("raw_data:latest")

    data = pd.read_csv(artifact.file())

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    data[column_name].value_counts().plot(kind="bar")

    ax.set_title(f"{column_name}")
    ax.set_xlabel("")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{column_name}_value_counts.png")

    wandb.log({f"{column_name}_value_counts": wandb.Image(f"{column_name}_value_counts.png")})
    wandb.finish()

def nltk_download():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

def get_lower_case():
    wandb.init(project="tweet_classifying", save_code=True)

    artifact = wandb.use_artifact("raw_data:latest")

    data = pd.read_csv(artifact.file(), usecols=['id','keyword', 'location', 'text', 'subset', 'target'])

    data = get_lower_text(data)

    data.to_csv("data_lower.csv", index=False)

    wandb.finish()

def remove_punctuation():
    data = pd.read_csv("data_lower.csv")

    data["text"] = data["text"].apply(punctuations)

    data.to_csv("data_no_punctuation.csv", index=False)

def tokenize_text():
    data = pd.read_csv("data_no_punctuation.csv")

    data["text_tokenized"] = data["text"].apply(tokenization)

    data.to_csv("data_tokenized.csv", index=False)

def remove_stopwords():
    stoplist = set(stopwords.words('english'))

    stoplist.remove('not')

    data = pd.read_csv("data_tokenized.csv")

    data["text_stop"] = data["text_tokenized"].apply(lambda x: stopwords_remove(x, stoplist))

    data.to_csv("data_no_stopwords.csv", index=False)

def lemma_text():
    lemmatizer = WordNetLemmatizer()

    data = pd.read_csv("data_no_stopwords.csv")

    data["text_lemmatized"] = data["text_stop"].apply(lambda x: lemmatization(lemmatizer, x))

    data.to_csv("data_lemmatized.csv", index=False)

def preprocessed_data():
    wandb.init(project="tweet_classifying", save_code=True)

    artifact = wandb.Artifact(
        name="preprocessed_data.csv",
        type="clean_data",
        description="Data after preprocessing"
    )

    data = pd.read_csv("data_lemmatized.csv")

    data["final"] = data["text_lemmatized"].str.join(" ")

    data.to_csv("preprocessed_data.csv", index=False)
    artifact.add_file("preprocessed_data.csv")
    wandb.run.log_artifact(artifact)
    wandb.finish()

def run_data_checks():
    result = pytest.main(["-vv", "."])

    if result != 0:
        raise ValueError("Data checks failed")

def segregate_data(subset):
    wandb.init(project="tweet_classifying", save_code=True)

    artifact = wandb.use_artifact("preprocessed_data.csv:latest")
    data = pd.read_csv(artifact.file())
    data = data[data["subset"] == subset].drop(columns=["subset"])

    data.to_csv(f"{subset}_data.csv", index=False)

def create_artifact(subset, artifact_type):
    wandb.init(project="tweet_classifying", save_code=True)

    artifact = wandb.Artifact(
        name=f"{subset}_data",
        type=artifact_type,
        description=f"{subset} data"
    )

    artifact.add_file(f"{subset}_data.csv")
    wandb.run.log_artifact(artifact)
    wandb.finish()

def create_data_to_train():
    wandb.init(project="tweet_classifying", job_type="train")

    artifact = wandb.use_artifact("train_data:latest")

    data = pd.read_csv(artifact.file())

    X = data['final']
    y = data['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

    X_train.to_csv("X_train.csv", index=False)
    X_val.to_csv("X_val.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_val.to_csv("y_val.csv", index=False)

    wandb.finish()

def create_dataset(dataset_type):
    X = pd.read_csv(f"X_{dataset_type}.csv")
    y = pd.read_csv(f"y_{dataset_type}.csv")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    encodings = tokenizer(list(X["final"].values), truncation=True, padding=True, return_tensors="tf")
    encodings.pop("token_type_ids", None) 

    dataset = tf.data.Dataset.from_tensor_slices((
        dict(encodings),
        tf.constant(y["target"].values, dtype=tf.int32)
    ))

    dataset = dataset.batch(4)

    tf.data.Dataset.save(dataset, f"{dataset_type}_dataset")

def train_model():
    train_dataset = tf.data.Dataset.load("train_dataset")
    val_dataset = tf.data.Dataset.load("val_dataset")

    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    tracker = EmissionsTracker(log_level="critical")
    tracker.start()

    history = model.fit(train_dataset, epochs=1, validation_data=val_dataset)

    emissions = tracker.stop()

    logging.info("{} kWh of electricity used since the begining".format(tracker.final_emissions_data.energy_consumed))
    logging.info("Energy consumed for RAM: {} kWh".format(tracker.final_emissions_data.ram_energy))
    logging.info("Energy consumed for all GPU: {} kWh".format(tracker.final_emissions_data.gpu_energy))
    logging.info("Energy consumed for all CPU: {} kWh".format(tracker.final_emissions_data.cpu_energy))
    logging.info("CO2 emission {}(in Kg)".format(tracker.final_emissions_data.emissions))

    wandb.init(project="tweet_classifying", save_code=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.plot(np.arange(0, 1), history.history["loss"], label="train_loss",linestyle='--')
    ax.plot(np.arange(0, 1), history.history["val_loss"], label="val_loss",linestyle='--')
    ax.plot(np.arange(0, 1), history.history["accuracy"], label="train_acc")
    ax.plot(np.arange(0, 1), history.history["val_accuracy"], label="val_acc")

    ax.set_title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss/Accuracy")

    ax.legend()
    plt.tight_layout()

    plt.savefig(f"training_loss_acc.png")

    wandb.log({f"training_loss_acc": wandb.Image(f"training_loss_acc.png")})
    wandb.finish()


WANDB_API_KEY = os.getenv("WANDB_API_KEY")

DEFAULT_ARGS = {
    "owner": "airflow",
    "start_date": datetime(2023, 11, 25),
    "catchup": False,
}


with DAG("classifying_tweets", default_args=DEFAULT_ARGS, schedule_interval="@daily") as dag:
    with TaskGroup("Fetching_Data"):
        download_train_data = PythonOperator(
            task_id="download_train_data",
            python_callable=download_data,
            op_kwargs={
                "url": "https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv",
                "file_name": "train.csv"
            },
        )

        download_test_data = PythonOperator(
            task_id="download_test_data",
            python_callable=download_data,
            op_kwargs={
                "url": "https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/test.csv",
                "file_name": "test.csv"
            },
        )

        wandb_login = PythonOperator(
            task_id="wandb_login",
            python_callable=wandb_login,
            op_kwargs={
                "api_key": WANDB_API_KEY
            },
        )

        create_subset_in_train_data = PythonOperator(
            task_id="create_subset_in_train_data",
            python_callable=create_subset,
            op_kwargs={
                "file_name": "train.csv"
            },
        )

        create_subset_in_test_data = PythonOperator(
            task_id="create_subset_in_test_data",
            python_callable=create_subset,
            op_kwargs={
                "file_name": "test.csv"
            },
        )

        concatenate_data = PythonOperator(
            task_id="concatenate_data",
            python_callable=concatenate_data,
        )

        wandb_artifact_upload = PythonOperator(
            task_id="wandb_artifact_upload",
            python_callable=wandb_artifact_upload,
        )

    delete_train_data = PythonOperator(
        task_id="delete_train_data",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "train.csv"
        },
    )

    delete_test_data = PythonOperator(
        task_id="delete_test_data",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "test.csv"
        },
    )

    delete_concatenated_data = PythonOperator(
        task_id="delete_concatenated_data",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "data_concatenated.csv"
        },
    )
    
    with TaskGroup("Exploratory_Data_Analysis"):
        artifact_download = PythonOperator(
            task_id="artifact_download",
            python_callable=wandb_artifact_download,
        )

        plot_target_value_counts = PythonOperator(
            task_id="plot_target_value_counts",
            python_callable=create_value_counts_plot,
            op_kwargs={
                "column_name": "target"
            },
        )

        plot_subset_value_counts = PythonOperator(
            task_id="plot_subset_value_counts",
            python_callable=create_value_counts_plot,
            op_kwargs={
                "column_name": "subset"
            },
        )

    with TaskGroup("Data_Preprocessing"):
        download_nltk = PythonOperator(
            task_id="dowload_nltk_data",
            python_callable=nltk_download,
        )

        data_lower_case = PythonOperator(
            task_id="data_lower_case",
            python_callable=get_lower_case,
        )

        data_no_punctuation = PythonOperator(
            task_id="data_no_punctuation",
            python_callable=remove_punctuation,
        )

        data_tokenized = PythonOperator(
            task_id="data_tokenized",
            python_callable=tokenize_text,
        )

        data_no_stopwords = PythonOperator(
            task_id="data_no_stopwords",
            python_callable=remove_stopwords,
        )

        data_lemmatized = PythonOperator(
            task_id="data_lemmatized",
            python_callable=lemma_text,
        )

        preprocessed_data = PythonOperator(
            task_id="preprocessed_data",
            python_callable=preprocessed_data,
        )

    delete_data_lower_case = PythonOperator(
        task_id="delete_data_lower_case",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "data_lower.csv"
        },
    )

    delete_data_no_punctuation = PythonOperator(
        task_id="delete_data_no_punctuation",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "data_no_punctuation.csv"
        },
    )

    delete_data_tokenized = PythonOperator(
        task_id="delete_data_tokenized",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "data_tokenized.csv"
        },
    )

    delete_data_no_stopwords = PythonOperator(
        task_id="delete_data_no_stopwords",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "data_no_stopwords.csv"
        },
    )

    delete_data_lemmatized = PythonOperator(
        task_id="delete_data_lemmatized",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "data_lemmatized.csv"
        },
    )

    delete_preprocessed_data = PythonOperator(
        task_id="delete_preprocessed_data",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "preprocessed_data.csv"
        },
    )

    with TaskGroup("Data_Checks"):
        run_data_tests = PythonOperator(
            task_id="run_data_checks",
            python_callable=run_data_checks,
        )

    with TaskGroup("Data_Segregation"):
        segregate_train_data = PythonOperator(
            task_id="segregate_train_data",
            python_callable=segregate_data,
            op_kwargs={
                "subset": "train"
            },
        )

        segregate_test_data = PythonOperator(
            task_id="segregate_test_data",
            python_callable=segregate_data,
            op_kwargs={
                "subset": "test"
            },
        )

        create_train_artifact = PythonOperator(
            task_id="create_train_artifact",
            python_callable=create_artifact,
            op_kwargs={
                "subset": "train",
                "artifact_type": "TrainData"
            },
        )

        create_test_artifact = PythonOperator(
            task_id="create_test_artifact",
            python_callable=create_artifact,
            op_kwargs={
                "subset": "test",
                "artifact_type": "TestData"
            },
        )

    delete_train_data_2 = PythonOperator(
        task_id="delete_train_data_2",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "train_data.csv"
        },
    )

    delete_test_data_2 = PythonOperator(
        task_id="delete_test_data_2",
        python_callable=delete_data,
        op_kwargs={
            "file_name": "test_data.csv"
        },
    )

    with TaskGroup("Model_Training"):
        get_data_to_train = PythonOperator(
            task_id="get_data_to_train",
            python_callable=create_data_to_train,
        )

        create_train_dataset = PythonOperator(
            task_id="create_train_dataset",
            python_callable=create_dataset,
            op_kwargs={
                "dataset_type": "train"
            },
        )

        create_val_dataset = PythonOperator(
            task_id="create_val_dataset",
            python_callable=create_dataset,
            op_kwargs={
                "dataset_type": "val"
            },
        )

        train_sequence_model = PythonOperator(
            task_id="train_sequence_model",
            python_callable=train_model,
        )

download_train_data.set_downstream(create_subset_in_train_data)
download_test_data.set_downstream(create_subset_in_test_data)
concatenate_data.set_upstream([create_subset_in_train_data, create_subset_in_test_data])
concatenate_data.set_downstream([wandb_login, delete_train_data, delete_test_data])
wandb_login.set_downstream([wandb_artifact_upload])
wandb_artifact_upload.set_downstream([artifact_download, delete_concatenated_data])

artifact_download.set_downstream([plot_target_value_counts, plot_subset_value_counts])

download_nltk.set_upstream([plot_target_value_counts, plot_subset_value_counts])
download_nltk.set_downstream(data_lower_case)
data_lower_case.set_downstream(data_no_punctuation)
data_no_punctuation.set_downstream([data_tokenized, delete_data_lower_case])
data_tokenized.set_downstream([data_no_stopwords, delete_data_no_punctuation])
data_no_stopwords.set_downstream([data_lemmatized, delete_data_tokenized])
data_lemmatized.set_downstream([preprocessed_data, delete_data_no_stopwords])
preprocessed_data.set_downstream([delete_data_lemmatized, delete_preprocessed_data, run_data_tests])

run_data_tests.set_downstream([segregate_train_data, segregate_test_data])
segregate_train_data.set_downstream([create_train_artifact])
segregate_test_data.set_downstream([create_test_artifact])
create_train_artifact.set_downstream([delete_train_data_2])
create_test_artifact.set_downstream([delete_test_data_2])

get_data_to_train.set_upstream(create_train_artifact)
get_data_to_train.set_downstream([create_train_dataset, create_val_dataset])
train_sequence_model.set_upstream([create_train_dataset, create_val_dataset])