FROM python:3.13-slim

# Build deps for pyarrow/numpy wheels are not needed on slim for these
# versions, but curl is useful for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

COPY config/ ./config/
COPY tests/ ./tests/

# Snapshots and runtime state are bind-mounted so they survive a rebuild:
# a pinned snapshot baked into an image would defeat the point of pinning.
VOLUME ["/app/data/snapshots", "/app/state"]

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# No default command: every entry point is explicit in docker-compose.yml,
# so nothing can start trading because a container was run bare.
CMD ["python", "-c", "print('Specify a command: see docker-compose.yml')"]
