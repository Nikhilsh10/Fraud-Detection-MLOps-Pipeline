# ---------------------------------------------------------------------------
# Fraud Detection API — Docker Image
# ---------------------------------------------------------------------------
# Multi-stage build:
#   stage 1 (builder) — install Python deps into a virtual environment
#   stage 2 (runtime) — copy venv + source; run uvicorn
#
# Build:
#   docker build -t fraud-api:latest .
#
# Run (standalone, requires mlruns/ volume for model):
#   docker run -p 8000:8000 \
#     -v "$(pwd)/mlruns:/app/mlruns" \
#     -v "$(pwd)/mlruns.db:/app/mlruns.db" \
#     fraud-api:latest
#
# Or use docker-compose (recommended):
#   docker-compose up --build
# ---------------------------------------------------------------------------

# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=100

# Copy dependency spec first (layer-cache friendly)
COPY pyproject.toml ./
# Minimal stub so pip can install the package in editable mode
COPY src/__init__.py ./src/__init__.py

# Create venv and install all production dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core deps declared in pyproject.toml [project.dependencies]
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        "scikit-learn==1.5.0" \
        "imbalanced-learn>=0.12" \
        "mlflow>=2.12" \
        "pandas>=2.2" \
        "numpy>=1.26" \
        "shap>=0.45" \
        "python-dotenv>=1.0" \
        "fastapi>=0.111" \
        "uvicorn[standard]>=0.29" \
        "pydantic>=2.7" \
        "boto3>=1.34"


# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/ ./src/
COPY pyproject.toml ./

# Pre-create directories that will be bind-mounted or written to at runtime
RUN mkdir -p mlruns reports data/raw data/reference

# Install the package itself (editable-equivalent for Docker)
RUN pip install --no-cache-dir --no-deps -e .

# ---------------------------------------------------------------------------
# Environment defaults (override via docker-compose or -e flags)
# ---------------------------------------------------------------------------
ENV MLFLOW_TRACKING_URI=sqlite:///mlruns.db
ENV PREDICTIONS_DB_PATH=/app/predictions.db
ENV DRIFT_LOG_PATH=/app/reports/drift_log.jsonl

# Expose the API port
EXPOSE 8000

# Health check — the /health endpoint must return 200
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c \
        "import urllib.request, sys; \
         r = urllib.request.urlopen('http://localhost:8000/health', timeout=4); \
         sys.exit(0 if r.status == 200 else 1)"

# Launch uvicorn
CMD ["uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
