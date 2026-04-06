# ── Email Triage OpenEnv — Dockerfile ──────────────────────────
# Compatible with Hugging Face Spaces (runs as user 1000, port 7860)

FROM python:3.11-slim

# Labels for HF Spaces discoverability
LABEL org.opencontainers.image.title="Email Triage OpenEnv"
LABEL org.opencontainers.image.description="Real-world email triage environment for AI agent training"
LABEL tags="openenv,email,triage,nlp"

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make sure the server package is importable
RUN touch server/__init__.py

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# HF Spaces uses port 7860 by default
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
