FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what the API needs at runtime: config + feature engineering + api + model
COPY config.py features.py ./
COPY api/ api/
COPY models/best_model.joblib models/

RUN useradd --create-home appuser
RUN mkdir logs && chown appuser logs   # API writes predictions.jsonl here
USER appuser

EXPOSE 8000
# PORT is injected by the host (Render); falls back to 8000 locally.
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health')" || exit 1
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
