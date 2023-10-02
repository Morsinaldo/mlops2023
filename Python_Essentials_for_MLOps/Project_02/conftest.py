import pytest
from dags.podcast import podcast_summary

# instantiate podcast_summary as a fixture
@pytest.fixture
def airflow_dag():
    return podcast_summary()

# fixture to simulate fake data
@pytest.fixture
def podcast_data():
    return [
        {
            "link": "https://example.com/episode1",
            "title": "Episode 1",
            "pubDate": "2023-09-28",
            "description": "Description 1",
            "enclosure": {"@url": "https://example.com/episode1.mp3"}
        },
        {
            "link": "https://example.com/episode2",
            "title": "Episode 2",
            "pubDate": "2023-09-29",
            "description": "Description 2",
            "enclosure": {"@url": "https://example.com/episode2.mp3"}
        },
    ]