FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV MODEL_DIR=/app/Model \
    INPUT_DIR=/app/Input \
    OUTPUT_DIR=/app/Comets_Output \
    WEB_DIR=/app/web \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

EXPOSE 8000

CMD ["/bin/sh", "-c", "uvicorn Api.main:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8000}"]
