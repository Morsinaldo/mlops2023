# Python Essentials for MLOps

Antes de iniciar no fabuloso mundo das operações de Machine Learning (MLOps), existem um conjunto de boas práticas a serem seguidas que facilitam a legibilidade e a manutenção do código. Afinal, em um projeto real com um modelo em produção, caso um erro aconteça, outros desenvolvedores irão ler o seu código para poder identificar o problema. Uma identação padronizada e loggings são muito úteis neste ponto. Ou ainda, se você precisa retreinar o modelo e a base de dados precisa ser tratada, verificar se não há nenhuma coluna inválida, se o número de instâncis é suficiente, entre outros. Nesse sentido, testes unitários podem facilitar bastante este processo. 

Com isso em vista, foram desenvolvidos três projetos com o intuito de explorar algumas dessas melhores práticas, as quais serão explicadas em mais detalhes abaixo.

## Projetos Desenvolvidos
   - [Video System Recomendation](./Project_01/): Inserir descrição
   - [Airflow Data Pipeline to Download Podcasts](./Project_02/): Inserir descrição
   - [Project 03](./Project_03/): Inserir descrição


## Melhores práticas

### Virtual Environment

Hoje em dia, utiliza-se muito cloud computing para hospedar aplicações e isso possui um custo associado. Então, quanto menos dependências a aplicação tiver, melhor. Nesse sentido, a criação de um ambiente virtual simula uma "caixa de areia" que contém as dependências que irão ser necessárias para a aplicação executar da forma correta. Além disso, a criação desse ambiente evita que uma bilbioteca específica de um projeto apresente conflito com uma bilbioteca de outro projeto, por exemplo.

Para tratar isso, existem alguns gerenciadores de pacotes que facilitam o tratamento desses conflitos. Como sugestão, eu apresento o [Anaconda](https://anaconda.org), o [Poetry](https://python-poetry.org) e o [Python Venv](https://docs.python.org/3/library/venv.html).

### Command Line Interface

A Interface de Linha de Comando (CLI) é o meio pelo qual nós executamos os scripts que estamos desenvolvendo e uma das possibilidades que esse meio abre é a passagem de parâmetros para a execução. Por exemplo, você pode colocar um parâmetro que define quais passos do seus pipeline o script vai executar, qual a proporção da divisão entre treinamento e teste ou até mesmo hiperparâmetros para o treinamento. O limite é a sua imaginação. :D

Em geral, quando estamos aprendendo machine learning, tendemos a aprender utilizando Jupyter Notebooks. Eles são bastante interessantes quando estamos fazendo experimentações e, apesar de algumas plataformas como Amazon SageMaker permitirem o deploy de aplicações utilizando Jupyter Notebooks, isso não é tão comum e nem é tão eficiente. Portanto, uma boa prática é desenvolver scripts que possam ser executados por meio do CLI.

Leitura recomendada: Data Science at the Command Line - Jeroen Jansses [1]

### Clean Code Principles

Uma vez que concordamos em escrever scripts, qual a melhor maneira de fazer isso? Não é apenas sair escrevendo o código como se não houvesse amanhã. Existem algumas boas práticas que podem ser seguidas para escrever códigos mais limpos, como por exemplo as apresentadas nos livros Clean Code in Python e Fluent Python:

1. Writing clean codes: utilizar *meaningful names* evitando abreviações e single letters. Além disso, pode-se ainda indicar o tipo da variável e utilizar nomes que descrevam da forma mais breve possível o significado daquela variável, função etc. Remember that long names are not the same as descriptive names.
2. Writing modular codes: Don't repeat yourself (DRY). Abstract out logic to improve readability. Function do one thing. Um código modularizado também facilita os testes unitários.
3. Refactoring: Sometimes you can modularize more your code or even reduce the execution time of your code improving internal structure without changing external functionality. For example, if you have a nested loop for and you can reduce the complexity using list comprehenssion or dictionaries.
4. Efficient code and documentation: Knowing how to write code that runs efficiently is another essential skill in software development. Note that one point is related to the other. The example above also applies to this one. When you're performing lots of different transformations on large amounts of data, this can make orders of magnitudes of difference in performance.
5. Documentation: É fundamental para quem está lendo o seu código, pois como uma pessoa que nunca viu o seu código vai poder olhar e entender de forma simples o que você fez? Comentários concisos de uma linha podem ser muito úteis para explicar o que aquele comando faz e docstrings para funções e classes utilizando a anotação dos tipos dos dados agilizam muito o entendimento de qual o objetivo daquela entidade.
6. Following PEPs: Python Enhancement Proposals (PEP) é um conjunto de diretrizes e recomendações para escrever código Python de maneira mais clara, legível e consistente. Ele define um estilo de codificação que ajuda os desenvolvedores a produzirem um código mais organizado e de fácil compreensão. Destacam-se:

   - [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
   - [PEP 3107 – Function Annotations](https://peps.python.org/pep-3107/)
   - [PEP 0484 – Type Hints](https://peps.python.org/pep-0484/)
   - [PEP 0526 – Syntax for Variable Annotations](https://peps.python.org/pep-0526/)
   - [PEP 0557 – Data Classes](https://peps.python.org/pep-0557/)
   - [PEP 0585 – Type Hinting Generics In Standard Collections](https://peps.python.org/pep-0585/)

Legal! Isso vai ajudar muito outras pessoas, mas como eu posso tornar esse processo mais rápido? Felizmente, já existem algumas bibliotecas que checam o nosso código em busca de inconformidades com a norma e informa o que não está de acordo. Esse processo é chamado de Linting 

### Linting

Nesse sentido, existem várias ferramentas que podem nos auxiliar na escrita de códigos padronizados. A primeira delas é [autopep8](https://pypi.org/project/autopep8/) e, de forma geral, ela automatiza a formatação do nosso código para o padrão do PEP8, realizando as mudanças diretamente no arquivo. Já o [pylint]() é uma biblioteca que analisa e gera uma nota de 0 a 10 para o nosso código e informa o que não está de acordo com a norma para o próprio usuário realize as modificações. O [pycodestyle](https://pypi.org/project/pycodestyle/) é bem similar ao pylint com a diferença de que ele não gera uma nota, apenas indica as modificações. As ferramentas de linting serão mostradas em mais detalhes no contexto de cada projeto.

### Tratamento de Erros

Este ponto é muito importante, pois no cenário de produção muito erros podem acontecer e nós devemos estar aptos e mitigar isso o mais rápido possível. Por exemplo, alguns erros que podem acontecer são:

   - Data Loading Failure: Imagine um modelo que é retreinado todos os dias com os novos injetados do dia anterior. Um belo dia, a fonte do dado muda sem aviso prévio e o pipeline não consegue lidar com essa mudança de forma adequada. Como resultado, o modelo não é treinado naquele dia, impactando em várias outras dowstream applications.
   - API Rate Limiting: Suponha que seu pipeline realiza o download dos dados de uma API externa. Caso a API impuser uma limitação de taxa e seu sistema não tratar essa exceção, o pipeline pode falhar ou ficar preso em um loop infinito, causando uma série de falhas downstream.
   - Resource Exhaustion: Imagine um modelo de aprendizado de máquina que realiza recomendação de produtos em um site de comércio eletrônico. Se o modelo apresentar um erro que não foi tratado adequadamente em um horário de pico de vendas, isso resultará em uma perda significativa de receita.
   - Timeout Errors: Poor handling of timeout errors can cascade, causing a failure in multiple dependent systems. Quanto mais completo o pipeline, maior o prejuízo que poderá ser causado.

EM face disso, o python fornece uma [documentação](https://docs.python.org/3/tutorial/errors.html) sobre este ponto, tanto os tratamentos mais padrões quanto tratamentos customizados.


### Pytest

Com o código padronizado, documentado e funcionando para o meu caso de uso, estamos prontos para colocar em produção? Calma, o seu código pode estar funcionando perfeitamente para o seu caso de uso, mas e se acontecerem os edge cases, como o seu código vai se comportar? An edge case is a problem or situation that occurs only at an extreme (maximum or minimum) operating parameter.

Uma ótima forma de tratar isso é com testes, em particupar o python possui uma ferramenta chamada [Pytest](https://docs.pytest.org/en/7.4.x/) que oferece várias features para facilitar esse processo. Uma delas é a Fixture, a qual permite instanciar um objeto de um dataset ou de um modelo, por exemplo, e utilizá-lo em vários testes, sem precisar instanciá-lo em cada um.

Essa prática faz com que você "gaste" um pouco mais de tempo pensando nos erros que podem acontecer, mas também faz com que você economize muito mais tempo no futuro procurando pelo o erro no seu código. Claro que é humanamente impossível pensar em todos os casos que podem acontecer, mas você pode começar se perguntando, por exemplo: 

- "Se minha aplicação está esperando receber um arquivo csv, o que acontece se ela receber um xlsx ou parquet?" 
- "Se minha função está esperando receber um dataset para fazer um processamento em uma determinada coluna, o que acontece se a coluna vier nula?"

Faça o teste e trate isso no código para que o teste dê certo! Quando você começa a pensar nessas situações, você começa a entrar numa inércia e vai pensando em cada vez mais situações. :D

### Logging

Once the ML has been deployed, it need to be monitored. Uma das formas mais elegantes de se fazer isso é utilizando a biblioteca [logging](https://docs.python.org/3/library/logging.html), uma vez que ela permite que o usuário defina diferentes níveis de logging, além do formato da mensagem, se o usuário deseja salvar em arquivo ou apenas mostrar no terminal etc. Isso nos permite, posteriormente, filtrar as mensagens por um determinado nível ou mesmo filtrar por uma determinada data, por exemplo. Dentre os níveis disponíveis, estão:

   - DEBUG: detailed information for debugging purposes
   - INFO: general confirmations that things are working as expected 
   - WARNING: an indication that something unexpected happened or may happen soon but the program still works as expected 
   - ERROR: due to a severe problem, the software has not been able to perform some function
   - CRITICAL: a very severe error that will likely lead to the application terminating
 
As práticas acima são apenas a ponta do iceberg nesse mundo de MLOps. Existem muitas que podem ser incorporadas dependendo do objetivo do seu projeto, da cultura da sua empresa, entre outras coisas. Como sugestão de leitura, caso você queira se aprofundar neste tema, destacam-se as seguintes referências:
   - Clean Architecture in Python
   