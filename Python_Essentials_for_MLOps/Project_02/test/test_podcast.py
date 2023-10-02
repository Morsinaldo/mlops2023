# Teste a função get_episodes
def test_get_episodes(podcast_data):
    episodes = get_episodes(podcast_data)
    assert len(episodes) == 2
    assert episodes[0]["title"] == "Episode 1"

# Teste a função load_episodes
def test_load_episodes(podcast_data, airflow_dag):
    # Execute a DAG no modo de teste
    airflow_dag.run(start_date="2023-09-30", end_date="2023-09-30", donot_pickle=True)

    # Aqui você pode testar a função load_episodes para verificar se os episódios foram carregados no banco de dados SQLite
    pass

# Teste a função download_episodes
def test_download_episodes(podcast_data, tmpdir, airflow_dag):
    tmp_dir = tmpdir.mkdir("episodes")

    # Execute a DAG no modo de teste
    airflow_dag.run(start_date="2023-09-30", end_date="2023-09-30", donot_pickle=True)

    audio_files = download_episodes(podcast_data, tmp_dir)
    assert len(audio_files) == 2
    assert os.path.exists(os.path.join(tmp_dir, "episode1.mp3"))