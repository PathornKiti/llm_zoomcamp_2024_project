FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install poetry


COPY pyproject.toml poetry.lock ./


RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

COPY . .

