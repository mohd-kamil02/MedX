# Shared by api, worker, and beat — same image, different commands.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY api/ ./api/
COPY ai/ ./ai/
COPY workers/ ./workers/

# Never run as root.
RUN useradd --create-home --uid 10001 medx && chown -R medx:medx /app
USER medx

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
