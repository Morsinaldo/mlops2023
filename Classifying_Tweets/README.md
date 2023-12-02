# Classifying Disaster-Related Tweets as Real or Fake

## Big picture

Neste projeto, foi desenvolvido um pipeline utilizando o Apache Airflow integrado com o **[Weights & Biases (Wandb)](https://wandb.ai/site)** para o treinamento de um modelo de aprendizado profundo baseado em Transformer com o intuito de classificar quais tweets são sobre desastres reais e quais não são. Utilizou-se a solução do **[Dataquest](https://github.com/dataquestio/solutions/blob/master/Mission797Solutions.ipynb)** como base para aplicar as melhores práticas de machine learning operations (MLOps), bem como os princípios vistos no **[primeiro projeto](../Python_Essentials_for_MLOps)**.

<!-- Além disso, foi utilizado o Apache Airflow para criar uma Direct Acyclic Graph (DAG) que automatiza a execução dos scripts e, por fim, tudo isso foi integrado à plataforma [Weights & Biases (Wandb)](https://wandb.ai/site) para realizar o rastreamento dos artefatos gerados ao longo da execução e das métricas obtidas. -->

O conjunto de dados é proveninete do **[Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/overview)**, em particular, da competição Natural Language Processing with Disaster Tweets, a qual tem como objetivo ser uma porta de entrada para cientistas de dados que querem desbravar o mundo de NLP. Nesse sentido, o objetivo é realizar processamentos na coluna `text` de forma a melhorar a predição dos modelos na classificação do tweet. Ao todo, o dataset possui as seguintes colunas: 

- `id` - a unique identifier for each tweet
- `text` - the text of the tweet
- `location` - the location the tweet was sent from (may be blank)
- `keyword` - a particular keyword from the tweet (may be blank)
- `target` - in `train.csv` only, this denotes whether a tweet is about a real disaster (1) or not (0)

A figura abaixo ilusta o passo-a-passo que será executado pela DAG, onde em laranja têm-se uma etapa executada e, em amarelo, um artefato gerado pela respectiva etapa. Cada artefato gerado em um passo é então passado para o próximo passo, como uma linha de montagem. Nesse sentido, o pipeline é composto por sete passos, os quais serão detalhados nas sub-seções a seguir.

![alt text](./images/Way.png)

## Pipeline

O **[Apache Airflow](https://airflow.apache.org)** foi utilizado para criar uma DAG que automatiza a execução dos scripts de maneira que tudo seja integrado à plataforma **[Weights & Biases (Wandb)](https://wandb.ai/site)** para realizar o rastreamento dos artefatos gerados ao longo da execução e das métricas obtidas. A figura abaixo ilustra a DAG criada para este projeto:

![alt text](./images/airflow_1.png)

### 1 - Fetch Data

<!-- O primero passo para todo projeto de machine learning ou data science é realizar o download dos dados. Então, é possível automatizar esse processo com um script que puxa os dados da fonte, seja uma API, um banco de dados, web scraping, entre outros. No contexto deste projeto, foi utilizado o comando `wget` para baixar os dados do Kaggle e depois utilizou-se o Wandb para armazenar estes dados. O código resumido pode ser conferido abaixo. -->

O primeiro passo é puxar os dados de algum lugar, seja via API, banco de dados, web scraping entre outros. Nesse sentido, foram criadas tasks para baixar os dados do Kaggle, criar uma nova coluna chamada `subset`, concatenar os dois conjuntos de dados e fazer o upload para o wandb.

![alt text](./images/airflow_2.png)

### 2 - Exploratory Data Analysis (EDA)

<!-- Nesta etapa, você pode realizar a automação da EDA por meio da geração de logs das distribuição estatística das colunas do dataset, por exemplo, ou gerar gráficos para poder realizar visualizações mais complexas posteriormente. Então, para fazer isso, realiza-se o download dos dados provenientes do passo anterior, realiza-se as análises e, por fim, sobe os artefatos para o Wandb. Abaixo, encontra-se um código resumo que faz o processo mencionado anteriormente e envia um gráfico em formato de imagem para o Wandb. -->

O próximo passo é realizar a análise exploratória de dados (EDA), a qual está representada neste passo pelas tasks que plotam os gráficos relacionados à distribuição estatística dos rótulos e da proporção entre treinamento e teste.

![alt text](./images/airflow_3.png)

<!-- É importante destacar que isso é bastante útil quando as colunas e os tipos dos seus dados já são bem conhecidos. -->

### 3 - Preprocessing

Nesta etapa, busca-se processar os dados de forma a deixá-los prontos para serem treinados e testados. Nesse sentido, os processamentos realizados variam muito de acordo com a natureza do problema. No contexto desse projeto, o texto passou pelos seguintes processamentos:

1. Lowercase
2. Removing punctuations and numbers
3. Tokenization
4. Removing Stopwords
5. Lemmatization

Os processamentos descritos foram aplicados sobre os dados brutos baixados na primeira etapa, como mostrado no código resumido abaixo. Todas as funções auxiliares utilizadas encontram-se no arquivo [preprocessing_helper.py](./preprocessing_helper.py)

![alt text](./images/airflow_4.png)

### 4 - Data Checks

<!-- Uma coisa interessante de ser monitorada nos modelos de aprendizado de máquina é a distribuição estatística dos dados que eles foram treinados e dos dados que ele está prevendo para evitar o data drift. Com isso em vista, esta é de suma importância para se fazer testes dessa natureza, bem como testar se os nomes e os tipos das colunas estão corretos, entre outros. Para fins didáticos, foram realizados três testes sobre os dados limpos para verificar se a coluna dos rótulos possuíam os valores esperados, se o dataset possuía pelo menos 1000 linhas e se a coluna do texto é do tipo object. O código resumido encontra-se abaixo. -->

Antes de treinar ou retreinar um modelo, é importante verificar a distribuição estatística dos conjuntos de dados para evitar o data drift. Então, neste passo foram realizados três testes sobre os dados limpos para verificar se a coluna dos rótulos possuíam os valores esperados, se o dataset possuía pelo menos 1000 linhas e se a coluna do texto é do tipo object. A task mostrada na imagem abaixo dispara todos os testes de uma vez.

![alt text](./images/airflow_5.png)

### 5 - Data Segregation 

É importante realizar a divisão entre os dados de treinamento e teste para que não ocorra o vazamento de dados. Então, esta é dedicada é realizar a divisão do conjunto de dados limpo entre as duas categorias mencionadas de forma que o dataset de teste fique totalmente isolado do de treinamento. As tasks realizam a divisão e fazem o upload os artefatos gerados.

![alt text](./images/airflow_6.png)

**Important**: Nesse caso, o conjunto de dados já possuía uma coluna que informa que a porção de treinamento e a porção de teste. Contudo, de forma geral, você pode utilizar a função `train_test_split` da biblioteca [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

### Train and validation

Uma vez que os dados estão limpos e estão divididos, é possível passar para o modelo realizar o treinamento. No contexto deste projeto, foi utilizado o `TFAutoModelForSequenceClassification`, um modelo baseado na arquitetura transformers do HuggingFace. Nesse sentido, as tasks abaixo pegam o conjunto de treinamento e divide-o entre treinamento e validação. 

![alt text](./images/airflow_7.png)

Em relação a solução original, foram realizados outros incrementos como a geração de um gráfico com as curvas de acurácia e perda, geração da matriz de confusão e o rastreamento da quantidade de CO2 emitida na atmosfera, bem como os gastos dos recursos computacionais durante o treinamento do modelo. Todas as essas informações foram sumarizadas e enviadas para o Wandb após a finalização desta etapa. O código resumido encontra-se abaixo:

### 7 - Test

Por fim, têm-se a etapa de teste do modelo treinado com o conjunto de dados que ele nunca viu. Contudo, este passo ainda não encontra-se disponível, pois a competição do Kaggle ainda não terminou. Então, os rótulos reais do conjunto de teste ainda não estão disponíveis.

<!-- Por mim, esta etapa realiza o carregamento do modelo salvo na etapa anterior e realiza as predições no dataset de teste salvo no passo 5. Com isso, é possível ter uma noção mais realística da performance do modelo, tendo em vista que ele está prevendo dados que ele nunca viu antes. No momento da escrita deste README, ainda não é possível se ter o feedback concreto da performance do nosso modelo, pois a competição do Kaggle ainda não terminou e, portanto, não temos os rótulos reais do dataset de teste. Ainda assim o código para realizar a inferência já foi desenvolvido e a versão resumida pode ser encontrada abaixo:

```python
# Initialize the W&B run
run = wandb.init(project="tweet_classifying", job_type="test")

# Load test dataet and separate between X and y
...

# Load the model
...

# Calculate predictions
logging.info("Infering")
tokenized = tokenizer(list(X_test), return_tensors="np", padding="longest")
outputs = model(tokenized).logits
classifications = np.argmax(outputs, axis=1)

# Evaluation Metrics
...

# Finish the run
run.finish()
``` -->

## Getting Started

### Prerequisites

- Wandb account
- Docker

### How to execute

1 - Clone this repoitory.

```bash
git clone https://github.com/Morsinaldo/mlops2023.git
```

2 - Enter in `Classifying_Tweets` folder.

```bash
cd Classifying_Tweets
```

3 - Create a [Weights & Biases](https://wandb.ai/) account.

4 - Go to [API Keys](https://wandb.ai/authorize) page and copy your key.

5 - Paste the generated key into the `WANDB_API_KEY` field in the [Dockerfile](./Dockerfile).

```Dockerfile
ENV WANDB_API_KEY = "your_key"
```

6 - Build the cointainer

```bash
docker compose up --build
```

7 - Access `http://localhost:8080` and put the following login.

```bash
Username: airflow
Password: airflow
```

After that, DAG will be available in the main page. You can trigger it manually.

## Results

O gráfico da acurácia e da perda tanto do conjunto de treinamento quanto do de validação está ilustrada na figura abaixo.

![alt text](./images/train_valid_loss_acc.png)

A matriz de confusão resultante está ilustrada na figura abaixo.

![alt text](./images/confusion_matrix.png)

As métricas obtidas estão abaixo:

- Train Accuracy: 0.9780
- Validation Accuracy: 0.8116
- Train Loss: 0.0542
- Validation Loss: 0.8966

- Energy consumed for RAM: 0.000589 kWh
- Energy consumed for all GPU: 0.008069 kWh
- Energy consumed for all CPU: 0.005263 kWh
- CO2 emission 0.006301 (in Kg)

É importante destacar que essas figuras ilustram o treinamento realizado no Google Colaboratory, por questões de tempo de treinamento. No pipeline do Airflow, configurou-se o treinamento apenas de uma época em CPU para fins de validação.

## Copyrights

This project was adapted from a `Guided Project` on the [Dataquest](https://www.dataquest.io/) website. Compared to the [original solution](https://github.com/dataquestio/solutions/blob/master/Mission797Solutions.ipynb), it involved transforming a Jupyter notebook into Python scripts to facilitate and enable the use with the Apache Airflow.

## References

- [Ivanovitch's Repository](https://github.com/ivanovitchm/mlops)
- [Build a Movie Recommendation System in Python (Dataquest)](https://github.com/dataquestio/solutions/blob/master/Mission797Solutions.ipynb)
