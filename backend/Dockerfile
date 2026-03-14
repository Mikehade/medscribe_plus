# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
# Install all Python dependencies into a virtual environment so the final
# image only copies what is needed (no build tools, no pip cache).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System packages needed to compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    poppler-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# Minimal image — only copies the venv and application code.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime system packages only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    ffmpeg \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /venv /venv

# Copy application source
COPY --chown=appuser:appgroup . .

# Create directories the app writes to at runtime
RUN mkdir -p /app/chroma_db /app/temp_uploads /app/logs && \
    chown -R appuser:appgroup /app/chroma_db /app/temp_uploads /app/logs

USER appuser

# Make the venv the default Python
ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Healthcheck — hits the /health endpoint every 30 s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
CMD ["python", "main.py"]