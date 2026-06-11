# ML Journey

Мой учебный репозиторий по Machine Learning / Data Science с упором на:

- сильную базу в Python
- работу с данными через Pandas и NumPy
- EDA (Exploratory Data Analysis)
- классическое машинное обучение
- практику с Git
- постепенный переход к реальным проектам, API и продакшен-инструментам

## Цель

Собрать учебный, но приближенный к реальной разработке ML-проект: от анализа данных и обучения модели до сохранения pipeline, inference API, тестов и Docker-упаковки.

Долгосрочная цель — выйти на уровень strong intern / junior специалиста в направлении Python + Data + ML Engineering.

## Текущий статус

Сейчас в репозитории реализован первый ML service skeleton на датасете Diabetes:

- обучение `RandomForestRegressor` pipeline;
- сохранение модели через `joblib`;
- inference script;
- FastAPI endpoint `/predict`;
- healthcheck endpoint `/health`;
- Pydantic validation;
- smoke tests через `pytest` и `TestClient`;
- Dockerfile для запуска API в контейнере;
- pre-commit hooks через Ruff.

## Текущий план обучения

### Phase 1 — Data
- Python
- NumPy
- Pandas
- EDA
- визуализация

### Phase 2 — Classical ML
- scikit-learn
- regression
- classification
- metrics
- cross-validation
- feature engineering

### Phase 3 — Projects
- 2–3 сильных проекта
- GitHub portfolio
- оформление репозиториев и README

### Phase 4 — ML Service
- FastAPI
- Docker
- базовый deployment

## Структура проекта

```text
ml-journey/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── diabetes_rf_pipeline.joblib
├── notebooks/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py
│   └── ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── inference.py
│       ├── modeling.py
│       ├── predict_diabetes_rf.py
│       └── train_diabetes_rf.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Local usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train model:

```bash
python -m src.ml.train_diabetes_rf
```

Run prediction script:

```bash
python -m src.ml.predict_diabetes_rf
```

Run API:

```bash
python -m uvicorn src.api.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

## Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest -q
```

Run pre-commit checks:

```bash
pre-commit run --all-files
```

## Docker usage

Build image:

```bash
docker build -t diabetes-ml-api .
```

Run container:

```bash
docker run -p 8000:8000 diabetes-ml-api
```

Open health endpoint:

```text
http://127.0.0.1:8000/health
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Example prediction payload:

```json
{
  "age": 0.0,
  "sex": 0.0,
  "bmi": 0.0,
  "bp": 0.0,
  "s1": 0.0,
  "s2": 0.0,
  "s3": 0.0,
  "s4": 0.0,
  "s5": 0.0,
  "s6": 0.0
}
```

Stop container:

```bash
docker ps
docker stop <container_id>
```
