# Sentinel FastAPI serving image — Day 4 Phase 3.
# Single-stage image. Day 7 graduates this to a multi-stage build with a
# slim runtime and a separate model-pull step.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OS deps: build-essential for xgboost/lightgbm wheels on slim, libgomp for the
# OpenMP runtime XGBoost needs at predict time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Day-4 Phase-3 deps that aren't in the pinned requirements.txt yet — kept
# here so the docker image is self-contained.
RUN pip install \
      "fastapi>=0.115,<0.120" \
      "uvicorn[standard]>=0.34,<0.40" \
      "pydantic>=2.5,<3" \
      "sqlalchemy>=2,<3" \
      "psycopg2-binary>=2.9,<3"

COPY src/ ./src/
COPY params.yaml ./params.yaml
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
