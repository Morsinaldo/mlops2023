"""
Main file of Movie Recomendation System
Author: Morsinaldo Medeiros
Date: 2023-09-26
"""
# import libraries
import os
import argparse
import logging
import tqdm
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import clean_movie_title, get_most_similar_movies_by_title, find_similar_movies

# set the logging level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# read the data
try:
    if not os.path.exists("data"):
        logging.info("Downloading the data")

        # use tqdm to show the progress bar
        r = requests.get("https://files.grouplens.org/datasets/movielens/ml-25m.zip",
                         stream=True, timeout=5)
        total_size = int(r.headers.get("content-length", 0))
        BLOCK_SIZE = 1024
        t = tqdm.tqdm(total=total_size, unit="iB", unit_scale=True)

        with open("data/ml-25m.zip", "wb") as f:
            for data in r.iter_content(BLOCK_SIZE):
                t.update(len(data))
                f.write(data)

        t.close()
    else:
        raise Exception("The data has already been downloaded")

except Exception as e:
    logging.error("Download failed")
    logging.error(e)

# create the parser
parser = argparse.ArgumentParser(description="Movie Recommendation System")

# add the arguments
parser.add_argument("--movie_title", type=str, help="The title of the movie")

# parse the arguments
args = parser.parse_args()

# get the movie title
movie_title = args.movie_title

# import the data
logging.info("Importing the data")
movies_df = pd.read_csv("data/movies.csv")

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
ratings_df = pd.read_csv("data/ratings.csv")

results = find_similar_movies(movies_df, ratings_df, movie_title)

# print the results
for index, row in results.iterrows():
    print(row["title"], row["genres"])
    print("")
