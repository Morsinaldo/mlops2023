import ktrain
import mlflow
import logging

import pandas as pd
from ktrain import text

# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

def train(artifact_folder: str):

    with mlflow.start_run():

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
        (x_train,y_train), (x_val,y_val), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                       x_test=X_test, y_test=y_test,
                                                                       class_names=class_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=512, 
                                                                       max_features=20000)

        # load model
        logger.info("Loading the model...")
        model = text.text_classifier('bert', train_data=(x_train,y_train), preproc=preproc)

        # wrap model and data in ktrain.Learner object
        logger.info("Wrapping the model...")
        learner = ktrain.get_learner(model, train_data=(x_train,y_train), 
                                    val_data=(x_val,y_val),
                                    batch_size=6)

        # fit the model
        logger.info("Fitting the model...")
        learner.fit_onecycle(2e-5, 1)

        # evaluate the model
        logger.info("Evaluating the model...")
        logger.info(learner.validate(val_data=(x_val,y_val), class_names=class_names))

        # save the model
        predictor = ktrain.get_predictor(learner.model, preproc)
        predictor.save("bert_trained")

        # log the artifact
        logger.info("Logging the artifacts...")
        mlflow.log_artifact("bert_trained")

        mlflow.log_metric("val_loss", learner.history.history['val_loss'][0])