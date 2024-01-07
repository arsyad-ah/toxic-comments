FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt src/ ./

RUN pip install -r requirements.txt

ENTRYPOINT ["uvicorn", "main:app", "--reload", "--app-dir", "src"]