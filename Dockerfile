FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV and the embedding stack need these shared runtime libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 app \
    && chown -R app:app /app

USER app

EXPOSE 1003

CMD ["python", "-m", "uvicorn", "ForFakebook.EmbeddingModel:app", "--host", "0.0.0.0", "--port", "1003"]
