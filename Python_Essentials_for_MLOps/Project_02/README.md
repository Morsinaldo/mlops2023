# Airflow Data Pipeline to Download Podcasts

## Introduction 

Sem dados, modelos de aprendizado de máquina ficam impossibilitados de aprender sobre a tarefa na qual busca-se automatizar. Por isso, é de fundamental importância criar mecanismos que realizem a extração, processamento e armazenamento destes dados, processo conhecido como data pipeline. 

Uma das melhores ferramentas que auxliam nesse processo de criação e automatização de um data pipeline é o [Apache Airflow](https://airflow.apache.org/). Com isso em vista, este projeto visa o desenvolvimento de uma aplicação utilizando o Airflow para automatizar o processo de coleta de episódios de podcast, armazenamento de informações relevantes em um banco de dados SQlite, download dos episódios de áudio e transcrição dos episódios para texto. Essa automatização é configurada para ser executada diariamente. A imagem abaixo mostra um print dos passos que o código realizará retirado da plataforma do Airflow.

## The code

De forma geral, têm-se primeiramente a criação de uma DAG (Directed Acyclic Graph) nomeada "podcast_summary", a qual foi configurada para ser executada diariamente desde o dia 30 de Maio de 2022. Abaixo, o código mostra a criação da DAG e da tabela do banco que irá armazenar os dados dentro da função `podcast_summary()`, a qual é responsável por encapsular todas as etapas da DAG. O código completo pode ser encontrado em [podcast.py](./dags/podcast.py)

```python
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
```

De forma geral, ela contém uma série de passos que serão executados em sequência, definidos com o decorator `@task`. Então a primeira task definida foi a `get_episodes()`, a qual é responsável por realizar o download dos dados através de uma requisição GET na URL definida, transformar para o formato dicionário e retornar os episódios, os quais são então armazenados no banco de dados.

```python
def podcast_summary() -> None:

    # rest of the code

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
```

A próxima task é definida como `load_episodes()` e ela recebe como entrada os dados dos episódios obtidos na etapa anterior. A partir disso, ela realiza a conexão com o banco de dados SQLite, recupera os episódios já armazenados no banco de dados, verifica se há novos episódios comparando os links dos episódios obtidos com os já armazenados e, por fim, insere os novos episódios no banco de dados.

```python
def podcast_summary() -> None:

    # rest of the code

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
```

O último passo da DAG é chamado de `transcribe_episodes()` e é responsável por realizar a conexão com o banco de dados, recuperar os episódios que ainda não foram transcritos, carregar um modelo de transcrição de fala Vosk, realizar a transcrição dos episódios de áudio para texto em chunks e, por fim, atualizar o banco de dados com as transcrições dos episódios.

## How to execute

O primero passo é criar um ambiente virtual com no máximo a versão 3.10 do Python, pois o airflow ainda não oferece suporte para a 3.11. Neste caso, eu utilizei a versão 3.10 e você pode fazer isso utilizando o comando 
```
venv -m 
```

```
python movie_recomendation.py --movie_title "movie_title"
```

### Execution Example

Vamos tomar o exemplo que você deseja procurar por recomendações de filmes parecidos com Super Mario.

```
python movie_recomendation.py --movie_title "Super Mario"
```

O resultado no seu terminal deve ser algo semelhante a este resultado.

# Incluir imagem da execução

## How to add more movies

A cada dia, são lançados novos filmes e, com isso, é desejável manter a nossa base de dados atualizada não é? Então, se você deseja fazer isso sem compromenter a exeução correta do código, eu criei alguns testes que irão amenizar a chance de ocorrer algum problema durante a execução. 

No arquivo [conftest.py](./conftest.py), você irá encontrar uma fixture que irá carregar os datasets em memória para poderem ser utilizados nos testes. Já no arquivo [test_data.py](./test/test_data.py), você irá encontrar testes que verificam, por exemplo, se o tipo de cada coluna está de acordo com o esperado, se há um número mínimo de linhas desejável, se os datasets possuem as colunas com os respectivos nomes esperados, entre outros. Sinta-se livre para adicionar mais testes na sua aplicação, eles são muito importantes e podem economizar muito tempo de depuração dos erros.

Para executar os testes, basta você rodar o comando abaixo:

```
pytest
```

# Incluir imagem da execução do pytest

## Code style

Note que eu tentei deixar o nome das variáveis da forma mais legível possível, bem como os nomes dos testes. Isso deixa o código mais limpo e mais legível, melhorando assim a sua legibilidade. Em relação ao AutoPep8, eu consegui uma nota X/10. Não fique tão obcecado em alcançar a nota 10. É preciso ter bom censo e ser crítico em relação à algumas coisas como a quantidade de espaços na identação, pois dependendo da resolução da tela que você está olhando, dois espaços podem ser mais interessantes do que quatro. 

Para executar o AutoPep8, basta você rodar o seguinte comando:

```
Colocar o comando do AutoPep8
```

## Copyrights

Este projeto foi adaptado de um `Portfolio Project` do site [Dataquest](https://www.dataquest.io/). Em relação à [solução original](https://github.com/dataquestio/project-walkthroughs/blob/master/movie_recs/movie_recommendations.ipynb), foi realizada a adaptação de um Jupyter notebook com um sistema de recomendação interativo para scripts em python com o intuito de facilitar e viabilizar a utilização do Pylint, do AutoPep8 e da passagem de argumentos através da linha de comando.