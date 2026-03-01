FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install "numpy>=1.26.0,<2.0.0"
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/

RUN mkdir -p data/processed

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn src.deployment.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
