"""
Main DAG file for the podcast summary project.
Author: Morsinaldo Medeiros
Date: 2023-09-30
"""
# import libraries
import os
import json
import requests
import xmltodict
import logging

from airflow.decorators import dag, task
import pendulum
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.providers.sqlite.hooks.sqlite import SqliteHook

from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# set constants
PODCAST_URL = "https://www.marketplace.org/feed/podcast/marketplace/"
EPISODE_FOLDER = "/home/morsinaldo/Desktop/airflow_podcast/dags/episodes"
FRAME_RATE = 16000

# set the logging level
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# create a DAG
@dag(
    dag_id='podcast_summary',
    schedule_interval="@daily",
    start_date=pendulum.datetime(2022, 5, 30),
    catchup=False,
)
def podcast_summary() -> None:
    """This DAG extracts, processes, and stores podcast episodes.
    
    Returns:
        None
    """

    # Create the database table
    create_database: SqliteOperator = SqliteOperator(
        task_id='create_table_sqlite',
        sql=r"""
        CREATE TABLE IF NOT EXISTS episodes (
            link TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            published TEXT,
            description TEXT,
            transcript TEXT
        );
        """,
        sqlite_conn_id="podcasts"
    )

    @task()
    def get_episodes() -> dict:
        """
        Fetches podcast episodes from the RSS feed.

        Returns:
            episodes (dict): A dictionary containing podcast episode information.
        """
        try:
            # Download data
            data = requests.get(PODCAST_URL)

            # Raise an exception if the request is not succeed 
            data.raise_for_status() 

            # Transform to dict
            feed = xmltodict.parse(data.text)

            # Get episodes
            episodes = feed["rss"]["channel"]["item"]
            logging.info(f"Found {len(episodes)} episodes.")
            return episodes
        except Exception as e:
            logging.error(f"Error fetching podcast episodes: {str(e)}")
            raise

    podcast_episodes: dict = get_episodes()
    create_database.set_downstream(podcast_episodes)

    @task()
    def load_episodes(episode_data: dict) -> list:
        """
        Loads new podcast episodes into the database.

        Args:
            episode_data (dict): A dictionary containing podcast episode information.

        Returns:
            new_episodes (list): A list of new episode records.
        """
        try:
            # Connect to the database
            hook = SqliteHook(sqlite_conn_id="podcasts")

            # Get stored episodes
            stored_episodes = hook.get_pandas_df("SELECT * from episodes;")
            new_episodes = []

            # Check for new episodes
            for episode in episode_data:
                if episode["link"] not in stored_episodes["link"].values:
                    filename = f"{episode['link'].split('/')[-1]}.mp3"
                    new_episodes.append([episode["link"], episode["title"], episode["pubDate"], episode["description"], filename])

            # Insert new episodes into the database
            hook.insert_rows(table='episodes', rows=new_episodes, target_fields=["link", "title", "published", "description", "filename"])
            logging.info(f"Loaded {len(new_episodes)} new episodes.")
            return new_episodes
        except Exception as e:
            logging.error(f"Error loading episodes into the database: {str(e)}")
            raise

    new_episodes: list = load_episodes(podcast_episodes)

    @task()
    def download_episodes(audio_info: list) -> list:
        """
        Downloads podcast episodes as audio files.

        Args:
            audio_info (list): A list of dictionaries containing audio file information.

        Returns:
            audio_files (list): A list of dictionaries containing audio file information.
        """
        try:
            audio_files = []
            for episode in audio_info:
                name_end = episode["link"].split('/')[-1]
                filename = f"{name_end}.mp3"
                audio_path = os.path.join(EPISODE_FOLDER, filename)

                # Download the audio file if it doesn't exist
                if not os.path.exists(audio_path):
                    logging.info(f"Downloading {filename}")
                    audio = requests.get(episode["enclosure"]["@url"])
                    with open(audio_path, "wb+") as f:
                        f.write(audio.content)
                audio_files.append({
                    "link": episode["link"],
                    "filename": filename
                })
            logging.info(f"Downloaded {len(audio_files)} audio files.")
            return audio_files
        except Exception as e:
            logging.error(f"Error downloading audio files: {str(e)}")
            raise

    audio_files: list = download_episodes(podcast_episodes)

    @task()
    def transcribe_episodes(audio_files: list, new_episode_records: list) -> None:
        """
        Transcribes audio episodes to text and updates the database.

        Args:
            audio_files (list): A list of dictionaries containing audio file information.
            new_episode_records (list): A list of new episode records.

        Returns:
            None
        """
        try:
            # Connect to the database
            hook = SqliteHook(sqlite_conn_id="podcasts")

            # Get untranscribed episodes
            untranscribed_episodes = hook.get_pandas_df("SELECT * from episodes WHERE transcript IS NULL;")

            # Load the Vosk model for transcription
            model = Model(model_name="vosk-model-en-us-0.22-lgraph")
            rec = KaldiRecognizer(model, FRAME_RATE)
            rec.SetWords(True)

            for index, row in untranscribed_episodes.iterrows():
                logging.info(f"Transcribing {row['filename']}")
                filepath = os.path.join(EPISODE_FOLDER, row["filename"])
                mp3 = AudioSegment.from_mp3(filepath)
                mp3 = mp3.set_channels(1)
                mp3 = mp3.set_frame_rate(FRAME_RATE)

                step = 20000
                transcript = ""

                # Transcribe audio in chunks
                for i in range(0, len(mp3), step):
                    logging.info(f"Transcription progress: {i/len(mp3)}")
                    segment = mp3[i:i+step]
                    rec.AcceptWaveform(segment.raw_data)
                    result = rec.Result()
                    text = json.loads(result)["text"]
                    transcript += text

                # Update the database with the transcript
                hook.insert_rows(table='episodes', rows=[[row["link"], transcript]], target_fields=["link", "transcript"], replace=True)
                logging.info(f"Transcribed episode {row['filename']}")
        except Exception as e:
            logging.error(f"Error transcribing episodes: {str(e)}")
            raise

    # Uncomment this to try speech to text (may not work)
    # transcribe_episodes(audio_files, new_episodes)

summary = podcast_summary()