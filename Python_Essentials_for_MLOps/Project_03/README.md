# Predicting Heart Disease

## Big Picture
The World Health Organization (WHO) approximates that 17.9 million individuals succumb to cardiovascular diseases (CVDs) annually.

Numerous risk factors may be conducive to the onset of CVD in an individual, including an unhealthy diet, insufficient physical activity, or mental health disorders. The early identification of these risk factors in individuals has the potential to prevent a substantial number of premature deaths.

In this project, we will use the [Kaggle dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/) and build a K-Nearest Neighbors classifier to accurately predict the likelihood of a patient having a heart disease in the future.

## The code

No arquivo [main.py](./main.py), foi utilizada a arquitetura `try - except` para realizar a leitura do dataset, a análise exploratória dos dados (EDA) e a limpeza das features. As funcões auxiliares utilizadas no script encontram-se no arquivo [utils.py](./utils.py) e o código mencionado encontra-se abaixo.

```python
# read the data
try:
    df_heart_disease = pd.read_csv('./data/heart_disease_prediction.csv')
except Exception as e:
    logging.error("Error loading data")
    logging.error(e)

# EDA
try:
    logging.info("Starting EDA")
    if not os.path.exists("images"):
        os.makedirs("images")
    eda_heart_disease(df_heart_disease)
except Exception as e:
    logging.error("Error during EDA")
    logging.error(e)

# Data Cleaning
try:
    logging.info("Starting data cleaning")
    df_heart_disease = clean_data(df_heart_disease)
except Exception as e:
    logging.error("Error during data cleaning")
    logging.error(e)
```

Um vez que o conjunto de dados encontra-se limpo, foi realizada a seleção de features e a plotagem de um mapa de calor com a correlação das colunas. Com isso, foi realizada a divisão entre treinamento e teste numa proporção padrão de 80% para treinamento e 20% para teste, mas o usuário pode definir um valor diferente através do parâmetro `test_size` durante a execução do script na linha de comando. O código abaixo mostra a execução dos comandos mencionados.

```python
# Feature selection
df_heart_disease = pd.get_dummies(df_heart_disease, drop_first=True)
logging.info(f"Dataset after one-hot encoding \n {df_heart_disease.head()}")

# plot the correlation matrix
fig = plt.figure(figsize=(16, 15))
sns.heatmap(df_heart_disease.corr(), annot=True)
plt.savefig("./images/correlation_matrix.png")

# split the data into train and test
X = df_heart_disease.drop("HeartDisease", axis=1)
y = df_heart_disease["HeartDisease"]

logging.info("Splitting the data into train and test")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

# log the shape of the data
logging.info("The shape of the training data is %s", X_train.shape)
logging.info("The shape of the test data is %s", X_test.shape)
```

A seguir, instancia-se um objeto MinMaxSaler para realizar deixar os valores das colunas escalonados e realiza-se o treinamento de fato do modelo KNeighborsClassifier. Com o modelo treinado, realiza-se as predições a avaliação do modelo por meio da métrica de acurácia. Tal código está mostrado abaixo.

```python
# scale the data
logging.info("Scaling the data")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train the model
logging.info("Training the model")
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# make predictions
logging.info("Making predictions")
y_pred = knn.predict(X_test_scaled)

# evaluate the model
logging.info("Evaluating the model")
accuracy = accuracy_score(y_test, y_pred)
logging.info("The accuracy of the model is %s", accuracy)
```

Caso o usuário deseje, ele pode ainda realizar um hyperparameter tuning do modelo através de uma busca Bayesiana de parâmetros. Por padrão, essa opção está definida como Falsa, mas o usuário pode ativá-la definindo o parâmetro `hyperparameter_tuning` igual a True na execução do script através da linha de comando. o código que realiza esse procedimento está mostrado abaixo:

```python
if args.hyperparameter_tuning == True:
    # define the hyperparameters
    hyperparameters = dict(n_neighbors=[1, 3, 5, 7, 9, 11, 13, 15],
                           weights=["uniform", "distance"],
                           metric=["euclidean", "manhattan", "minkowski"])

    # define the search
    logging.info("Instantiating the search")
    search = BayesSearchCV(knn, hyperparameters, cv=5)

    # fit the search
    logging.info("Fitting the model")
    best_model = search.fit(X_train_scaled, y_train)

    # summarize best
    logging.info("Best accuracy: %s", best_model.best_score_)
    logging.info("Best hyperparameters: %s", best_model.best_params_)

    # make predictions
    logging.info("Making predictions")
    y_pred = best_model.predict(X_test_scaled)

    # evaluate the model
    logging.info("Evaluating the model")
    accuracy = accuracy_score(y_test, y_pred)
```

## How to execute

Uma vez que você já tenha criado o ambiente virtual de sua preferência e instalado as dependências, basta você ativá-lo e executar o comando abaixo.

```
python main.py
```

Para permitir uma maior flexibilidade da utilzação do script através da linha de comando, ele permite a passagem de dois parâmetros:

- test_size: permite que o usuário defina a proporção que será utilizada para o conjunto de teste durante o treinamento. Aceita valores do tipo float entre 0 e 1. Default: 0.2
- hyperparameter_tuning: permite que o usuário defina se deseja realizar a busca bayesiana pelo melhores hyperparâmetros do modelo. Aceita valores True or False. Default: False

### Execution Example

Supondo que você deseja uma proporção de 15% para teste e deseja realizar o hyperaparameter tuning

```
python main.py --test_size 0.15 --hyperparameter_tuning True
```

## Pytest

### How to add more data

A cada dia novos dados são gerados e, com isso, é desejável manter a nossa base de dados atualizada não é? Então, se você deseja fazer isso sem compromenter a exeução correta do código, eu criei alguns testes que irão amenizar a chance de ocorrer algum problema durante a execução. 

No arquivo [conftest.py](./conftest.py), você irá encontrar uma fixture que irá carregar o dataset em memória para poder ser utilizado nos testes. Já no arquivo [test_data.py](./test/test_data.py), você irá encontrar testes que verificam, por exemplo, se o tipo de cada coluna está de acordo com o esperado, se todas estão presentes, se as colunas categóricas possuem os valores esperados, se as colunas numéricas estão com valores dentro de um certo intervalo entre outros. Sinta-se livre para adicionar mais testes na sua aplicação, eles são muito importantes e podem economizar muito tempo de depuração dos erros.

Para executar os testes, basta você rodar o comando abaixo:

```
pytest
```

## Clean codes Principles

Note que, em relação a [solução original](https://github.com/dataquestio/solutions/blob/master/Mission740Solutions.ipynb), vários nomes de variáveis e de funções foram alterados de forma a deixar o código mais legível, além da modularização de alguns passos do pipeline e da adição do tratamento de exceções e da adição de loggings. Além disso, realizou-se a documentação das funções e, formato docstring e utilizando a dica de tipo, como mostra o exemplo da função abaixo:

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to replace the zero values in the RestingBP 
    and Cholesterol columns with the median of the respective group.

    Args:
        df (pd.DataFrame): The dataframe to be cleaned

    Returns:
        df_clean (pd.DataFrame): The cleaned dataframe
    """

    # only keep non-zero values for RestingBP
    df_clean = df[df["RestingBP"] != 0]
    logging.info(f"Dataset has {df_clean.shape[0]} non-zero values for RestingBP")

    # get the lines where HeartDisease is 0
    df_heartdisease_mask = df_clean["HeartDisease"]==0
    logging.info(f"Dataset has {df_heartdisease_mask.sum()} lines where HeartDisease is 0")

    # filter the lines where HeartDisease is 0
    df_cholesterol_without_heartdisease = df_clean.loc[df_heartdisease_mask, "Cholesterol"]
    df_cholesterol_with_heartdisease = df_clean.loc[~df_heartdisease_mask, "Cholesterol"]

    logging.info(f"Replace the zero values with the median of the respective group")
    # replace the zero values with the median of the respective group
    df_clean.loc[df_heartdisease_mask, "Cholesterol"] = df_cholesterol_without_heartdisease.replace(to_replace = 0, value = df_cholesterol_without_heartdisease.median())
    df_clean.loc[~df_heartdisease_mask, "Cholesterol"] = df_cholesterol_with_heartdisease.replace(to_replace = 0, value = df_cholesterol_with_heartdisease.median())

    logging.info(f"Describe of RestingDB and Cholesterol \n {df_clean.describe()}")

    return df_clean
```

## Code style

Em relação à estilização do código, com o AutoPep8, eu consegui uma nota X/10 no arquivo [main.py](./main.py). Não fique tão obcecado em alcançar a nota 10. É preciso ter bom censo e ser crítico em relação à algumas coisas como a quantidade de espaços na identação, pois dependendo da resolução da tela que você está olhando, dois espaços podem ser mais interessantes do que quatro. 

Para executar o AutoPep8, basta você rodar o seguinte comando:

```
Colocar o comando do AutoPep8
```

## Copyrights

Este projeto foi adaptado de um `Guided Project` do site [Dataquest](https://www.dataquest.io/), em particular, do caminho [Machine Learning in Python](https://app.dataquest.io/learning-path/machine-learning-in-python-skill). Em relação à [solução original](https://github.com/dataquestio/project-walkthroughs/blob/master/movie_recs/movie_recommendations.ipynb), foi realizada a adaptação de um Jupyter notebook para scripts em python com o intuito de facilitar e viabilizar a utilização do Pylint, do AutoPep8 e da passagem de argumentos através da linha de comando.
