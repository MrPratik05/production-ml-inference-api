import logging
import os
import time
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

try:
    from transformers import pipeline
except Exception:
    pipeline = None


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s"
)

MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

ERROR_COUNT = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

app = FastAPI(
    title="Production ML Inference API",
    description="A production-style ML inference service with CI/CD, Docker, logging, health checks and monitoring.",
    version="1.0.0"
)

model = None


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, example="I really enjoyed this product.")


class PredictionResponse(BaseModel):
    label: str
    score: float
    model_name: str
    latency_ms: float


@app.on_event("startup")
def load_model() -> None:
    global model
    if pipeline is None:
        logging.warning("Transformers is not available. Using fallback rule-based predictor.")
        model = None
        return

    logging.info("Loading model: %s", MODEL_NAME)
    model = pipeline("sentiment-analysis", model=MODEL_NAME)
    logging.info("Model loaded successfully.")


def fallback_predict(text: str) -> Dict[str, float]:
    positive_words = {"good", "great", "excellent", "amazing", "love", "liked", "happy", "enjoyed"}
    negative_words = {"bad", "poor", "terrible", "hate", "hated", "awful", "sad", "disappointed"}

    words = set(text.lower().split())
    pos_score = len(words.intersection(positive_words))
    neg_score = len(words.intersection(negative_words))

    if pos_score >= neg_score:
        return {"label": "POSITIVE", "score": 0.75 if pos_score else 0.55}
    return {"label": "NEGATIVE", "score": 0.75}


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "service": "production-ml-inference-api"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        with PREDICTION_LATENCY.time():
            if model is None:
                result = fallback_predict(payload.text)
            else:
                output = model(payload.text)[0]
                result = {
                    "label": output["label"],
                    "score": float(output["score"])
                }

        latency_ms = round((time.time() - start_time) * 1000, 2)

        logging.info(
            "Prediction completed | label=%s | score=%.4f | latency_ms=%.2f",
            result["label"],
            result["score"],
            latency_ms
        )

        return PredictionResponse(
            label=result["label"],
            score=result["score"],
            model_name=MODEL_NAME,
            latency_ms=latency_ms
        )

    except Exception as exc:
        ERROR_COUNT.inc()
        logging.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
