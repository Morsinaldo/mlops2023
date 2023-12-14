1 - Crie o ambiente virtual

```bash
conda env create -f environment.yml
```

2 - Abra um terminal e execute o server do mlflow

```bash
mlflow server --host 127.0.0.1 --port 5000
```

3 - Em outro terminal, execute o arquivo `main.py`

```bash
python main.py
```

4 - Execute o GradIO

```bash
python app.py
```