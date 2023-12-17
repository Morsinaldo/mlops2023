"""
Python file to preprocess the data.
"""

import re
import logging
import nltk
import mlflow
import pandas as pd

from textblob import Word
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from eda import get_raw_data_artifact

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download("reuters")
nltk.download('words')

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def preprocessing(artifact_folder: str):
    """
    Preprocess the data.

    Parameters
    ----------
    artifact_folder : str
        Folder to save the data.
    """
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment
    mlflow.set_experiment("Multiclass Text Classification")

    # get the raw data artifact
    logger.info("Getting the raw data artifact...")
    get_raw_data_artifact()
    logger.info("Raw data artifact downloaded successfully!")

    with mlflow.start_run(run_name="preprocessing"):
        logger.info("Starting Preprocessing...")

        # read raw data artifact
        logger.info("Reading the raw data...")
        try:
            df=pd.read_csv(f"{artifact_folder}/bbc-text.csv", engine='python', encoding='UTF-8')
        except pd.errors.DataError as e:
            logger.error(e)

        # Preprocessing
        logger.info("Preprocessing the data...")
        df['text']=df['text'].fillna("")
        logger.info(df.isna().sum())

        df['lower_case'] = df['text'].apply(
            lambda x: x.lower().strip().replace('\n', ' ').replace('\r', ' ')
        )

        df['alphabatic'] = df['lower_case'].apply(
            lambda x: re.sub(r'[^a-zA-Z\']', ' ', x)
        ).apply(
            lambda x: re.sub(r'[^\x00-\x7F]+', '', x)
        )

        df['without-link'] = df['alphabatic'].apply(
            lambda x: re.sub(r'http\S+', '', x)
        )

        tokenizer = RegexpTokenizer(r'\w+')

        df['Special_word'] = df.apply(
            lambda row: tokenizer.tokenize(row['lower_case']),
            axis=1
        )

        stop_list = [
            "my",
            "haven't",
            "aren't",
            "can",
            "no",
            "why",
            "through",
            "herself",
            "she",
            "he",
            "himself",
            "you",
            "you're",
            "myself",
            "not",
            "here",
            "some",
            "do",
            "does",
            "did",
            "will",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "should",
            "should've",
            "couldn't",
            "mightn't",
            "mustn't",
            "shouldn't",
            "hadn't",
            "wasn't",
            "wouldn't"
        ]

        stop = [word for word in stopwords.words('english') if word not in stop_list]

        df['stop_words'] = df['Special_word'].apply(
            lambda x: [item for item in x if item not in stop]
        )

        df['stop_words'] = df['stop_words'].astype('str')
        df['short_word'] = df['stop_words'].str.findall(r'\w{2,}')
        df['string'] = df['short_word'].str.join(' ')

        df['Text'] = df['string'].apply(
            lambda x: " ".join([Word(word).lemmatize() for word in x.split()])
        )

        # save the preprocessed data
        logger.info("Saving the preprocessed data...")
        df.to_csv(f"{artifact_folder}/bbc-text-preprocessed.csv", index=False)

        # log the artifact
        logger.info("Logging the artifacts...")
        mlflow.log_artifact(f"{artifact_folder}/bbc-text-preprocessed.csv")
