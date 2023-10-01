# Build a Movie Recommendation System in Python

## Big picture
Neste projeto, foi desenvolvido um sistema de recomendação de vídeos utilizando o Python utilizando algumas das melhores práticas utilizadas para tornar o seu código, mais limpo, mais legível e mais fácil de depurar. O sistema consiste em alguns scripts execuátáveis por meio da linha de comando passando como parâmetros, por exemplo, o título de um filme que você gosta e quantos itens semelhantes você deseja que o sistema te retorne. 

## The code

No arquivo [movie_recomendation.py](./movie_recomendation.py), foi utilizada a arquitetura `try - except` para verificar se o diretório dos dados já existe e depois para realizar o download propriamente dito dos dados a partir da url. Para isso, foi utilizada a biblioteca [tqdm](https://github.com/tqdm/tqdm) para mostrar uma barra de progresso no terminal para que o usuário não fique perdido sem saber uma estimativa do tempo do download ou mesmo se o programa ainda está executando como deveria. Caso ele não consiga realizar isso, o programa irá lançar uma exceção e informar para o usuário qual foi o erro.

```python
try:
    if not os.path.exists("data"):
        logging.info("Downloading the data")
        
        # use tqdm to show the progress bar
        r = requests.get("https://files.grouplens.org/datasets/movielens/ml-25m.zip", stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        t = tqdm.tqdm(total=total_size, unit="iB", unit_scale=True)

        with open("data/ml-25m.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)

        t.close()
    else:
        raise Exception("The data has already been downloaded")
          
except Exception as e:
    logging.error("Download failed")
    logging.error(e)
```

Com os dados no seu devido lugar, nós podemos instânciar o objeto que irá receber e processar os argumentos informados na linha de comando, o `ArgumentParser`. Neste ponto, será realizada uma verificação do formato do título informado, o carregamento do conjunto de dados, a limpeza dos caracteres especiais presentes nos títulos.

```python
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
logging.info(f"The shape of the data is {movies_df.shape}")

# clean the movie title
logging.info("Cleaning the movie title")
movies_df["clean_title"] = movies_df["title"].apply(clean_movie_title)
```

Em seguida, instânciou-se um `TfidfVectorizer` para transformar os títulos dos filmes em vetores e, com isso, poder buscar os filmes com os títulos mais parecidos. Ok, isso já é bem legal, mas vamos combinar que podemos fazer melhor do que é isso num é?! Que tal recomendar filmes similares com base no consumo e na nota de outros usários?

```python
# instantiate the TF-IDF vectorizer
logging.info("Instantiating the TF-IDF vectorizer")
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

# fit the vectorizer and transform the data
logging.info("Fitting and transforming the data")
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df["clean_title"])

# get the most similar movies
logging.info(f"Getting the most similar movies to {movie_title}")
results = get_most_similar_movies_by_title(movies_df, tfidf_vectorizer, movie_title)
```

No arquivo [ratings.csv](./data/ratings.csv), nós podemos encontrar a nota que cada usuário deu para um determinado `movieId` e, através dessa coluna, nós podemos pegar o respectivo nome do filme no arquivo [movies.csv](./data/movies.csv). A função responsável por realizar essa busca é definida como `find_similar_movies()` e pode ser encontrada no arquivo [utils.py](./utils.py).

```python
# movie_recomendation.py

# read the ratings data
ratings_df = pd.read_csv("data/ratings.csv")

results = find_similar_movies(movies_df, ratings_df, movie_title)
```

## How to execute

Uma vez que você já tenha criado o ambiente virtual de sua preferência e instalado as dependências, basta você ativá-lo e executar o comando abaixo substituindo "movie_title" pelo título do filme que você deseja.

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
