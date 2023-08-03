FROM python:3.10 as base

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN apt-get update && apt-get install -y libsndfile1 libsndfile1-dev ffmpeg

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# CMD uvicorn main:api --port 8008
CMD uvicorn main:api --host 0.0.0.0 --port 8008 --log-level info --root-path /api --workers 32
EXPOSE 8008 
