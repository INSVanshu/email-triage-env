
# Email Triage OpenEnv — Dockerfile
# Cache-bust: v4 — forces full rebuild with clamped scores

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Email Triage OpenEnv"
LABEL tags="openenv,email,triage,nlp"

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN touch server/__init__.py
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
