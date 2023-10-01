"""
Utils file of Movie Recomendation System
Author: Morsinaldo Medeiros
Date: 2023-09-26
"""
# import libraries
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_movie_title(raw_movie_title: str) -> str:
    """
    Clean the title of the movie by removing any non-alphanumeric characters
    
    Args:
        raw_movie_title (str): The raw title of the movie

    Returns:
        clean_movie_title (str): The clean title of the movie
    """
    clean_movie_title = re.sub("[^a-zA-Z0-9 ]", "", raw_movie_title)
    return clean_movie_title

def get_most_similar_movies_by_title(movies_df: pd.DataFrame, 
                                     tfidf_vectorizer: TfidfVectorizer,
                                     movie_title: str) -> pd.DataFrame:
    """
    Get the most similar movies by title
    
    Args:
        movies_df (pd.DataFrame): The dataframe containing the movies
        tfidf_vectorizer (TfidfVectorizer): The tfidf vectorizer
        movie_title (str): The title of the movie

    Returns:
        results (pd.DataFrame): The most similar movies
    """
    clean_title = clean_movie_title(movie_title)
    embbeding = tfidf_vectorizer.transform([clean_title])
    similarity = cosine_similarity(embbeding, tfidf_vectorizer.transform(movies_df["clean_title"])).flatten()
    most_similar_movie_indices = np.argpartition(similarity, -5)[-5:]
    results = movies_df.iloc[most_similar_movie_indices].iloc[::-1]

    return results

def find_similar_movies(movies_df: pd.DataFrame, 
                        ratings_df: pd.DataFrame, 
                        movie_title: str) -> pd.DataFrame:
    """
    Find similar movies based on the ratings of similar users
    
    Args:
        movies_df (pd.DataFrame): The dataframe containing the movies
        ratings_df (pd.DataFrame): The dataframe containing the ratings
        movie_title (str): The title of the movie

    Returns:
        rec_percentages (pd.DataFrame): The most similar movies
    """
    # get the movie id
    movie_id = movies_df[movies_df["title"] == movie_title]["movieId"].values[0]

    # get the similar users
    similar_users = ratings_df[(ratings_df["movieId"] == movie_id) & (ratings_df["rating"] > 4)]["userId"].unique()

    # get the similar users recommendations
    similar_user_recs = ratings_df[(ratings_df["userId"].isin(similar_users)) & (ratings_df["rating"] > 4)]["movieId"]

    # get the percentage of recommendations
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    # get all the users recommendations
    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings_df[(ratings_df["movieId"].isin(similar_user_recs.index)) & (ratings_df["rating"] > 4)]

    # get the percentage of recommendations
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    # merge the dataframes
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)

    # rename the columns
    rec_percentages.columns = ["similar", "all"]
    
    # get the score
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

    # sort the values
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    # get the top 10 recommendations
    return rec_percentages.head(10).merge(movies_df, left_index=True, right_on="movieId")[["score", "title", "genres"]]