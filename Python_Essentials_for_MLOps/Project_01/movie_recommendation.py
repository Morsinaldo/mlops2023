"""
Main file of Movie Recomendation System
Author: Morsinaldo Medeiros
Date: 2023-09-26
"""
# import libraries
import os
import zipfile
import argparse
import logging
import requests
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import clean_movie_title, get_most_similar_movies_by_title, \
                 find_similar_movies, download_data

# set the logging level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

load_dotenv()

# create the parser
parser = argparse.ArgumentParser(description="Movie Recommendation System")

# add the arguments
parser.add_argument("--movie-title", type=str, help="The title of the movie")

# parse the arguments
args = parser.parse_args()

# get the movie title
movie_title = args.movie_title

# read the data
try:
    if not os.path.exists("ml-25m"):
        logging.info("Downloading the data")

        download_data(os.getenv("URL"), os.getenv("ZIP_FILENAME"))
    else:
        logging.info("The data is already downloaded")
except requests.exceptions.ConnectionError:
    logging.error("Connection Error")
except requests.exceptions.Timeout:
    logging.error("Timeout Error")
except requests.exceptions.HTTPError:
    logging.error("HTTP Error")
except zipfile.BadZipFile:
    logging.error("The downloaded file is not a valid ZIP file.")

# import the data
logging.info("Importing the data")
movies_df = pd.read_csv("ml-25m/movies.csv")

# log the shape of the data
logging.info("The shape of the data is %s", movies_df.shape)

# clean the movie title
logging.info("Cleaning the movie title")
movies_df["clean_title"] = movies_df["title"].apply(clean_movie_title)

# instantiate the TF-IDF vectorizer
logging.info("Instantiating the TF-IDF vectorizer")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

# fit the vectorizer and transform the data
logging.info("Fitting and transforming the data")
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df["clean_title"])

# get the most similar movies
logging.info("Getting the most similar movies to %s", movie_title)
results = get_most_similar_movies_by_title(movies_df, tfidf_vectorizer, movie_title)

# print the results
for index, row in results.iterrows():
    print(row["title"], row["genres"])
    print("")

# read the ratings data
ratings_df = pd.read_csv("ml-25m/ratings.csv")

logging.info("Finding similar by user ratings")

results = find_similar_movies(movies_df, ratings_df, results.iloc[0]["title"])

# print the results
for index, row in results.iterrows():
    print(row["title"], row["genres"])
    print("")
