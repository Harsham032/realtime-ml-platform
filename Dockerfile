# Multi-stage build: compilers and headers stay in the builder, never reaching
# the runtime image.
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# LightGBM needs libgomp at build and run time; XGBoost ships its own.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies install before the source is copied so a code change does not
# invalidate the dependency layer.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-deps .


FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RTML_ENV=production

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 rtml

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=rtml:rtml configs ./configs
COPY --chown=rtml:rtml scripts ./scripts
COPY --chown=rtml:rtml data/README.md ./data/README.md

RUN mkdir -p /app/data/processed /app/artifacts /app/reports /app/mlartifacts \
    && chown -R rtml:rtml /app/data /app/artifacts /app/reports /app/mlartifacts

USER rtml
EXPOSE 8000

# The service answers /health even with no model loaded, reporting why, so the
# check distinguishes "not ready" from "not running".
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "rtml.services.api:app", "--host", "0.0.0.0", "--port", "8000"]
