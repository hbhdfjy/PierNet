FROM node:20-slim AS frontend

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM python:3.11-slim AS backend

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY piern ./piern
COPY scripts ./scripts
COPY configs ./configs
COPY api_server.py ./
COPY --from=frontend /app/frontend/dist ./frontend/dist

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e . --no-deps

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
