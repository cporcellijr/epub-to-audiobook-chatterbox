# Resemble Enhance service: a thin FastAPI wrapper around resemble-enhance.
# Weights are not baked in: they live in a volume (~1.6 GB) and download on first load.
FROM python:3.10-slim
RUN apt-get update && apt-get install --no-install-recommends -y git git-lfs libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir resemble-enhance fastapi uvicorn httpx pytest
WORKDIR /app
COPY *.py /app/
EXPOSE 8006
CMD ["python", "server.py"]
