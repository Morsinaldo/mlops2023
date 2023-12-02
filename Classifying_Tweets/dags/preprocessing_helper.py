import re
import nltk
import pandas as pd
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def get_lower_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'text' column in a DataFrame to lowercase.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with the 'text' column in lowercase.
    """
    df['text'] = df['text'].str.lower()
    return df

def punctuations(inputs: str) -> str:
    """
    Remove non-alphabetic characters from the input string.

    Parameters:
    - inputs (str): The input string.

    Returns:
    - str: The input string with non-alphabetic characters removed.
    """
    return re.sub(r'[^a-zA-Z]', ' ', inputs)

def tokenization(inputs: str) -> List[str]:
    """
    Tokenize the input string into individual words.

    Parameters:
    - inputs (str): The input string.

    Returns:
    - List[str]: List of tokens (words).
    """
    return word_tokenize(inputs)

def stopwords_remove(inputs: List[str], stoplist: set) -> List[str]:
    """
    Remove stopwords from the list of input tokens.

    Parameters:
    - inputs (List[str]): List of input tokens.
    - stoplist (set): Set of stopwords to be removed.

    Returns:
    - List[str]: List of tokens with stopwords removed.
    """
    return [k for k in inputs if k not in stoplist]

def lemmatization(lemmatizer: WordNetLemmatizer, inputs: List[str]) -> List[str]:
    """
    Lemmatize the list of input tokens using WordNet lemmatizer.

    Parameters:
    - lemmatizer (WordNetLemmatizer): WordNet lemmatizer object.
    - inputs (List[str]): List of input tokens.

    Returns:
    - List[str]: List of lemmatized tokens.
    """
    return [lemmatizer.lemmatize(word=kk, pos='v') for kk in inputs]
