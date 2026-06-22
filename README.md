# Production ML Inference API

A production-style machine learning inference service built with **FastAPI, Docker, GitHub Actions CI/CD, automated testing, logging, and Prometheus monitoring**.

This project is designed to demonstrate practical MLOps skills commonly required in ML Engineer, AI Engineer, Data Scientist, and Applied AI roles.

## Project Overview

The API serves a pre-trained Hugging Face sentiment analysis model through a REST endpoint. It includes production-oriented features such as:

- REST API for model inference
- Health check endpoint
- Prometheus-compatible monitoring endpoint
- Request count monitoring
- Prediction latency monitoring
- Error tracking
- Docker containerization
- Automated testing with Pytest
- CI/CD workflow using GitHub Actions
- Environment-based configuration
- Structured logging

## Architecture

```text
Client Request
    ↓
FastAPI Application
    ↓
Hugging Face Model
    ↓
Prediction Response
    ↓
Metrics + Logs
```

## Tech Stack

- Python
- FastAPI
- Hugging Face Transformers
- Docker
- GitHub Actions
- Pytest
- Prometheus Client
- Uvicorn

## API Endpoints

### Health Check

```bash
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "service": "production-ml-inference-api"
}
```

### Prediction

```bash
POST /predict
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I really enjoyed this product."}'
```

Example response:

```json
{
  "label": "POSITIVE",
  "score": 0.9998,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "latency_ms": 52.73
}
```

### Monitoring Metrics

```bash
GET /metrics
```

This exposes Prometheus-compatible metrics including:

- `prediction_requests_total`
- `prediction_errors_total`
- `prediction_latency_seconds`

## Run Locally

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## Run with Docker

```bash
docker build -t production-ml-inference-api .
docker run -p 8000:8000 production-ml-inference-api
```

Or with Docker Compose:

```bash
docker compose up --build
```

## Run Tests

```bash
pytest -q
```

## CI/CD

The GitHub Actions workflow runs automatically on push and pull requests to `main`.

It performs:

1. Code checkout
2. Python setup
3. Dependency installation
4. Automated test execution
5. Docker image build

Workflow file:

```text
.github/workflows/ci.yml
```

## Deployment Options

You can deploy this service to:

- Render
- Railway
- Fly.io
- Azure App Service
- AWS ECS
- Google Cloud Run

For a quick deployment, Render or Railway is easiest.

## Production Features Demonstrated

This project demonstrates the following production ML engineering practices:

- Model serving through an API
- Containerized deployment
- Automated CI/CD pipeline
- Unit and integration testing
- Health checks
- Runtime monitoring
- Error tracking
- Latency monitoring
- Environment variable configuration
- Logging for observability

## CV / Resume Bullet

**Production ML Inference API — FastAPI, Docker, CI/CD, Monitoring**

- Developed a production-style ML inference service using FastAPI and a pre-trained Hugging Face transformer model.
- Containerized the application with Docker and implemented automated CI/CD using GitHub Actions.
- Added health checks, structured logging, Prometheus-compatible metrics, request tracking, error counting, and latency monitoring.
- Built automated tests with Pytest and documented reproducible setup, API usage, deployment workflow, and monitoring endpoints.

## Suggested GitHub Repository Description

Production-style ML inference API with FastAPI, Docker, GitHub Actions CI/CD, automated testing, logging, and Prometheus monitoring.

## Suggested GitHub Topics

```text
mlops fastapi docker cicd github-actions prometheus monitoring machine-learning model-deployment huggingface
```
