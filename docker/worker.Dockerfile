FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONPATH=/app
ENV CELERY_WORKER_CONCURRENCY=1
ENV CELERY_WORKER_LOGLEVEL=info

CMD ["sh", "-c", "celery -A src.webapp.tasks worker --loglevel=${CELERY_WORKER_LOGLEVEL:-info} --concurrency=${CELERY_WORKER_CONCURRENCY:-1}"]
