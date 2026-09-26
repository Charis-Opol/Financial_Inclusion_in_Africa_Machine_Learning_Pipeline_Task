# syntax=docker/dockerfile:1
#
# Prediction service for the XGBoost/class-weight bank-account model.
#
# Base image choice: python:3.13-slim (Debian), multi-stage.
#   - The serving deps are numpy + xgboost-cpu + the web stack; all ship
#     manylinux wheels, so no compiler is needed and there is no case for a
#     CUDA/PyTorch base image (torch is a training-only dependency here).
#   - Not Alpine: musl libc can't use manylinux wheels, so numpy/xgboost would
#     have to be compiled from source -- slower builds, larger, riskier image.
#   - Matches the Python minor version the model was trained and tested on.
#   - Two stages keep pip, its cache and build metadata out of the runtime
#     image; the runtime stage receives only the finished virtualenv.
# For reproducible production builds, pin the base by digest
# (python:3.13-slim@sha256:...) once you've pulled a version you trust.

ARG PYTHON_IMAGE=python:3.13-slim

# ---- Stage 1: resolve and install dependencies into an isolated venv ----------
FROM ${PYTHON_IMAGE} AS deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
# The venv gets no pip of its own (-13 MB); the base image's pip installs into it.
RUN python -m venv --without-pip /opt/venv
# Copy ONLY the requirements file first: this layer (the slow one) is rebuilt
# only when dependencies change, not on every source or model edit.
COPY requirements-serving.txt /tmp/requirements-serving.txt
# --only-binary: fail the build rather than silently compile from source if a
# wheel is ever missing for this platform.
RUN pip --python /opt/venv/bin/python install --only-binary=:all: -r /tmp/requirements-serving.txt

# ---- Stage 2: minimal runtime --------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# libgomp1: XGBoost's OpenMP runtime. It's not in the slim image and without
# it `import xgboost` fails at startup.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Fixed, non-root UID/GID so volume ownership is predictable across rebuilds.
ARG APP_UID=10001
RUN groupadd --system --gid ${APP_UID} app \
    && useradd --system --uid ${APP_UID} --gid app --no-create-home --shell /usr/sbin/nologin app

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models/production \
    INFERENCE_THREADS=1 \
    LOG_LEVEL=INFO \
    PORT=8000

WORKDIR /app
COPY --from=deps /opt/venv /opt/venv

# Ordered least- to most-frequently changed: the model artifact changes only on
# retraining; source code changes on every edit.
COPY --chown=root:root models/production/ /app/models/production/
COPY --chown=root:root src/ /app/src/

# The prediction-log directory is the only path the app writes to. Creating it
# here, owned by `app`, means a fresh named volume mounted on it inherits that
# ownership (Docker copies image ownership into an empty named volume).
RUN mkdir -p /app/logs && chown app:app /app/logs
VOLUME ["/app/logs"]

USER app
EXPOSE 8000

# /health returns 503 when the model failed to load, so an unusable container
# is reported unhealthy. Uses Python's urllib because slim has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request, sys; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4).status == 200 else 1)"]

# One worker process per container: the model is CPU-bound and loaded per
# process, so scale by adding containers/replicas (each with its own CPU
# limit) rather than by forking workers inside one CPU-limited container.
# `exec` so uvicorn is PID 1 and receives SIGTERM for graceful shutdown.
CMD ["sh", "-c", "exec uvicorn fin_inclusion.serving.app:app --host 0.0.0.0 --port ${PORT}"]
