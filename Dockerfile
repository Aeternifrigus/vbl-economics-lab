# Runtime image for vbl-economics-lab (CLI only).
FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Dependencies first, from the hash-pinned lock, so this layer is cached until the lock changes.
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

# Then the package itself, without re-resolving dependencies.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# The reconciled raw data and claims ship in the image. Snapshots, models and reports are mounted.
COPY data/raw/ ./data/raw/
COPY claims/ ./claims/
RUN mkdir -p data/external models reports && chown -R appuser:appuser /app

USER appuser
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["pwl"]
CMD ["--help"]
