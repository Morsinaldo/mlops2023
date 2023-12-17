"""
Python file to run all the steps in the pipeline.
"""

from fetch_data import fetch_data
from eda import eda
from preprocessing import preprocessing
from data_checks import run_tests
from data_segregation import data_segregation
from train import train

def main():
    """
    Main function.
    """

    fetch_data(
        url="https://huggingface.co/datasets/SetFit/bbc-news/raw/main/bbc-text.csv",
        artifact_folder="artifacts"
    )

    eda(
        figures_folder="figures",
        artifact_folder="artifacts"
    )

    preprocessing(
        artifact_folder="artifacts"
    )

    run_tests()

    data_segregation(
        artifact_folder="artifacts"
    )

    train(
        artifact_folder="artifacts",
        figures_folder="figures"
    )

if __name__ == "__main__":
    main()
