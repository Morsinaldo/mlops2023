# Classifying Disaster-Related Tweets as Real or Fake

## Big picture

Neste projeto, foi treinado um modelo de aprendizado profundo baseado em Transformer com o intuito de classificar quais tweets são sobre desastres reais e quais não são. Utilizou-se a solução do [Dataquest](https://github.com/dataquestio/solutions/blob/master/Mission797Solutions.ipynb) como base para aplicar as melhores práticas de machine learning operations (MLOps), bem como os princípios vistos no [primeiro projeto](../Python_Essentials_for_MLOps). Além disso, foi utilizado o Apache Airflow para criar uma Direct Acyclic Graph (DAG) que automatiza a execução dos scripts e, por fim, tudo isso foi integrado à plataforma Weights & Biases (Wandb) para realizar o rastreamento dos artefatos gerados ao longo da execução e das métricas obtidas.

O conjunto de dados é proveninete do [Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/overview), em particular, da competição Natural Language Processing with Disaster Tweets, a qual tem como objetivo ser uma porta de entrada para cientistas de dados que querem desbravar o mundo de NLP. Nesse sentido, o objetivo é realizar processamentos na coluna `text` de forma a melhorar a predição dos modelos na classificação do tweet. Ao todo, o dataset possui as seguintes colunas: 

- `id` - a unique identifier for each tweet
- `text` - the text of the tweet
- `location` - the location the tweet was sent from (may be blank)
- `keyword` - a particular keyword from the tweet (may be blank)
- `target` - in `train.csv` only, this denotes whether a tweet is about a real disaster (1) or not (0)

A figura abaixo ilusta o passo-a-passo que será executado pela DAG, onde em laranja têm-se uma etapa executada e, em amarelo, um artefato gerado pela respectiva etapa. Nesse sentido, o pipeline é composto por sete passos, os quais serão detalhados nas sub-seções a seguir.

![alt text](./images/Way.png)

### 1 - Fetch Data

O primero passo para todo projeto de machine learning ou data science é realizar o download dos dados. Então, é possível automatizar esse processo com um script que puxa os dados da fonte, seja uma API, um banco de dados, web scraping, entre outros. No contexto deste projeto, foi utilizado o comando `wget` para baixar os dados do Kaggle e depois utilizou-se o Wandb para armazenar estes dados. O código resumido pode ser conferido abaixo.

```python
# download datasets
!wget https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv

# Login to Weights & Biases
!wandb login --relogin

# send the raw_data to the wandb and store it as an artifact
!wandb artifact put \
      --name tweet_classifying/raw_data \
      --type RawData \
      --description "Real and Fake Disaster-Related Tweets Dataset" raw_data.csv
```

### 2 - Exploratory Data Analysis (EDA)

Nesta etapa, você pode realizar a automação da EDA por meio da geração de logs das distribuição estatística das colunas do dataset, por exemplo, ou gerar gráficos para poder realizar visualizações mais complexas posteriormente. Então, para fazer isso, realiza-se o download dos dados provenientes do passo anterior, realiza-se as análises e, por fim, sobe os artefatos para o Wandb. Abaixo, encontra-se um código resumo que faz o processo mencionado anteriormente e envia um gráfico em formato de imagem para o Wandb.

```python
# Initialize wandb run
wandb.init(project='tweet_classifying', save_code=True)

# Get the artifact
artifact = wandb.use_artifact('raw_data:latest')

# Download the content of the artifact to the local directory
artifact_dir = artifact.download()

# read data
df = pd.read_csv(artifact.file())

# ploting figure
freq_target = df['target'].value_counts()

fig, ax = plt.subplots(figsize=(12,7))
freq_target.plot(kind='bar',alpha=1, rot=0, colormap=plt.cm.tab10)

plt.title('Tweet Count by Category', size=20)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.tick_params(top="off", left="off", right="off", bottom='off')
plt.savefig('target_distribution.png')  # Save the plot to a file
plt.show()
plt.close()

# Log the histogram image to wandb
wandb.log({"Target Distribution": wandb.Image('target_distribution.png')})

# finish run
wandb.finish()
```


É importante destacar que isso é bastante útil quando as colunas e os tipos dos seus dados já são bem conhecidos.

### 3 - Preprocessing

Nesta etapa, busca-se processar os dados de forma a deixá-los prontos para serem treinados e testados. Nesse sentido, os processamentos realizados variam muito de acordo com a natureza do problema. No contexto desse projeto, o texto passou pelos seguintes processamentos:

1. Lowercase
2. Removing punctuations and numbers
3. Tokenization
4. Removing Stopwords
5. Lemmatization

Os processamentos descritos foram aplicados sobre os dados brutos baixados na primeira etapa, como mostrado no código resumido abaixo. Todas as funções auxiliares utilizadas encontram-se no arquivo [preprocessing_helper.py](./preprocessing_helper.py)

```python
# Initialize wandb run
run = wandb.init(project='tweet_classifying', save_code=True)

# Get the artifact
artifact = wandb.use_artifact('raw_data:latest')

# Transform into a dataframe
...

# Lower Character all the Texts
df = get_lower_text(df)

# Removing Punctuations and Numbers from the Text
df['text'] = df['text'].apply(punctuations)

# Tokenize the text into individual words
df['text_tokenized'] = df['text'].apply(tokenization)

# Get the set of English stopwords
stoplist = set(stopwords.words('english'))
stoplist.remove('not')

# Remove Stopwords
df['text_stop'] = df['text_tokenized'].apply(lambda x: stopwords_remove(x, stop_words=stoplist))

# Instantiate Lemmatizer
lemmatizer = WordNetLemmatizer()

# Apply Lemmatizer
df['text_lemmatized'] = df['text_stop'].apply(lambda x: lemmatization(lemmatizer, x))

# Joining Tokens into Sentences
df['final'] = df['text_lemmatized'].str.join(' ')

# Save the Dataframe and log the artifact
...
```

### 4 - Data Checks

Uma coisa interessante de ser monitorada nos modelos de aprendizado de máquina é a distribuição estatística dos dados que eles foram treinados e dos dados que ele está prevendo para evitar o data drift. Com isso em vista, esta é de suma importância para se fazer testes dessa natureza, bem como testar se os nomes e os tipos das colunas estão corretos, entre outros. Para fins didáticos, foram realizados três testes sobre os dados limpos para verificar se a coluna dos rótulos possuíam os valores esperados, se o dataset possuía pelo menos 1000 linhas e se a coluna do texto é do tipo object. O código resumido encontra-se abaixo.

```python
def test_target_labels(data):
    # Ensure that the 'target' column has only 0 and 1 as labels, excluding NaN
    actual_labels = set(data['target'].unique())

    # Check for equality excluding NaN
    assert all(math.isnan(label) or label in {0.0, 1.0} for label in actual_labels)

def test_dataset_size(data):
    # Ensure that the dataset has at least 1000 rows
    assert len(data) >= 1000

def test_final_column_type(data):
    # Ensure that the 'final' column is of type string
    assert data['final'].dtype == 'O'
```

### 5 - Data Segregation 

É importante realizar a divisão entre os dados de treinamento e teste para que não ocorra o vazamento de dados. Então, esta é dedicada é realizar a divisão do conjunto de dados limpo entre as duas categorias mencionadas de forma que o dataset de teste fique totalmente isolado do de treinamento. A proporção da divisão pode ser ajustada diramente no código e uma versão resumida deste passo encontra-se abaixo:

```python
# Initialize wandb run
run = wandb.init(project='tweet_classifying', job_type='data_segregation')

# Get the clean_data artifact
artifact = run.use_artifact('preprocessed_data.csv:latest')

# Transform into a Dataframe
...

# Get the subsets by the label in subset column
train_data = df[df["subset"] == "train"].drop(["subset"], axis=1)
test_data = df[df["subset"] == "test"].drop(["subset"], axis=1)

# Save as csv file
...

# Log the artifact to wandb
run.log_artifact(train_artifact)
run.log_artifact(test_artifact)

# Finish the wandb run
wandb.finish()
```

**Important**: Nesse caso, o conjunto de dados já possuía uma coluna que informa que a porção de treinamento e a porção de teste. Contudo, de forma geral, você pode utilizar a função `train_test_split` da biblioteca [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).

### Train and validation

Uma vez que os dados estão limpos e estão divididos, é possível passar para o modelo realizar o treinamento. No contexto deste projeto, foi utilizado o `TFAutoModelForSequenceClassification`, um modelo baseado na arquitetura transformers do HuggingFace. Além do treinamento do modelo, foram realizados outros incrementos como a geração de um gráfico contendo as curvas de acurácia e custo tanto do dataset de treinamento quanto do de validação (20% do de treinamento), geração da matriz de confusão e o rastreamento da quantidade de CO2 emitida na atmosfera, bem como os gastos dos recursos computacionais durante o treinamento do modelo. Todas as essas informações foram sumarizadas e enviadas para o Wandb após a finalização desta etapa. O código resumido encontra-se abaixo:

```python
# Initialize the W&B run
run = wandb.init(project="tweet_classifying", job_type="train")

# Use W&B artifact for training data
train_file = run.use_artifact("train_data:latest").file()
df_train = pd.read_csv(train_file)

# Separating X and y
X = df_train['final']
y = df_train['target']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

# Load tokenizer and tokenize the text
...

# Create TensorFlow datasets and the create the batches
...

# Load the model
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5)
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Save model
model.save_pretrained("tweet_classifying_finetuning")

# Upload model artifact and the figures
...

# Finish the run
run.finish()
```

### 7 - Test

Por mim, esta etapa realiza o carregamento do modelo salvo na etapa anterior e realiza as predições no dataset de teste salvo no passo 5. Com isso, é possível ter uma noção mais realística da performance do modelo, tendo em vista que ele está prevendo dados que ele nunca viu antes. No momento da escrita deste README, ainda não é possível se ter o feedback concreto da performance do nosso modelo, pois a competição do Kaggle ainda não terminou e, portanto, não temos os rótulos reais do dataset de teste. Ainda assim o código para realizar a inferência já foi desenvolvido e a versão resumida pode ser encontrada abaixo:

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
```

## How to execute

## Results

## Copyrights

This project was adapted from a `Guided Project` on the [Dataquest](https://www.dataquest.io/) website. Compared to the [original solution](https://github.com/dataquestio/solutions/blob/master/Mission797Solutions.ipynb), it involved transforming a Jupyter notebook into Python scripts to facilitate and enable the use with the Apache Airflow.

## References

- [Ivanovitch's Repository](https://github.com/ivanovitchm/mlops)
- [Build a Movie Recommendation System in Python (Dataquest)](https://github.com/dataquestio/solutions/blob/master/Mission797Solutions.ipynb)
