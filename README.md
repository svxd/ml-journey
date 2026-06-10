# ML Journey

Мой учебный репозиторий по Machine Learning / Data Science с упором на:

- сильную базу в Python
- работу с данными через Pandas и NumPy
- EDA (Exploratory Data Analysis)
- классическое машинное обучение
- практику с Git
- постепенный переход к реальным проектам, API и продакшен-инструментам

## Цель

Выйти на собеседования как сильный стажёр / junior ML engineer или Data Scientist.

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
│   │   └── main.py
│   └── ml/
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── inference.py
│       ├── modeling.py
│       ├── predict_diabetes_rf.py
│       └── train_diabetes_rf.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Run ML API with Docker

Build Docker image:

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
