# syntax=docker/dockerfile:1

# ---- Builder stage -------------------------------------------------------
# python:3.11-slim (glibc) is required — sentence-transformers / torch wheels
# are not published for Alpine's musl libc.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build tooling needed by some wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch FIRST so the heavyweight CUDA build is never pulled in
# as a transitive dependency of sentence-transformers.
RUN pip install --prefix=/install torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- Runtime stage -------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# curl is needed for the container HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --create-home --uid 1000 aria

WORKDIR /app

# Copy installed Python packages from the builder, then the application code.
COPY --from=builder /install /usr/local
COPY app ./app
COPY config ./config

# Data directories for the persisted vector store and dedup state.
RUN mkdir -p /data/chroma /data/dedup && chown -R aria:aria /app /data

USER aria

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
