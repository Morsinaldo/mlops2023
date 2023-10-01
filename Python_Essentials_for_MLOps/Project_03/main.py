"""
Main file of Heart Disease Prediction
Author: Morsinaldo Medeiros
Date: 2023-10-01
"""
# import libraries
import os
import logging
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import eda_heart_disease, clean_data

from skopt import BayesSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# set the logging level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# create the parser
parser = argparse.ArgumentParser(description="Heart Disease Prediction")

# add the arguments
parser.add_argument("--test_size", type=float, default=0.2, help="The proportion of the dataset to include in the test split")
parser.add_argument("--hyperparameter_tuning", type=bool, default=False, help="Whether to perform hyperparameter tuning or not")

# parse the arguments
args = parser.parse_args()

# read the data
try:
    df_heart_disease = pd.read_csv('./data/heart_disease_prediction.csv')
except Exception as e:
    logging.error("Error loading data")
    logging.error(e)

# EDA
try:
    logging.info("Starting EDA")
    if not os.path.exists("images"):
        os.makedirs("images")
    eda_heart_disease(df_heart_disease)
except Exception as e:
    logging.error("Error during EDA")
    logging.error(e)

# Data Cleaning
try:
    logging.info("Starting data cleaning")
    df_heart_disease = clean_data(df_heart_disease)
except Exception as e:
    logging.error("Error during data cleaning")
    logging.error(e)

# Feature selection
df_heart_disease = pd.get_dummies(df_heart_disease, drop_first=True)
logging.info(f"Dataset after one-hot encoding \n {df_heart_disease.head()}")

# plot the correlation matrix
fig = plt.figure(figsize=(16, 15))
sns.heatmap(df_heart_disease.corr(), annot=True)
plt.savefig("./images/correlation_matrix.png")

# split the data into train and test
X = df_heart_disease.drop("HeartDisease", axis=1)
y = df_heart_disease["HeartDisease"]

logging.info("Splitting the data into train and test")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

# log the shape of the data
logging.info("The shape of the training data is %s", X_train.shape)
logging.info("The shape of the test data is %s", X_test.shape)

# scale the data
logging.info("Scaling the data")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train the model
logging.info("Training the model")
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# make predictions
logging.info("Making predictions")
y_pred = knn.predict(X_test_scaled)

# evaluate the model
logging.info("Evaluating the model")
accuracy = accuracy_score(y_test, y_pred)
logging.info("The accuracy of the model is %s", accuracy)

if args.hyperparameter_tuning == True:
    # define the hyperparameters
    hyperparameters = dict(n_neighbors=[1, 3, 5, 7, 9, 11, 13, 15],
                           weights=["uniform", "distance"],
                           metric=["euclidean", "manhattan", "minkowski"])

    # define the search
    logging.info("Instantiating the search")
    search = BayesSearchCV(knn, hyperparameters, cv=5)

    # fit the search
    logging.info("Fitting the model")
    best_model = search.fit(X_train_scaled, y_train)

    # summarize best
    logging.info("Best accuracy: %s", best_model.best_score_)
    logging.info("Best hyperparameters: %s", best_model.best_params_)

    # make predictions
    logging.info("Making predictions")
    y_pred = best_model.predict(X_test_scaled)

    # evaluate the model
    logging.info("Evaluating the model")
    accuracy = accuracy_score(y_test, y_pred)