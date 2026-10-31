# Runtime image for packaged-water-lab (CLI only).
FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Install the package from source (non-editable) so the image has no dependency on the build dir.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Data and outputs are mounted or copied in; ship the reconciled raw data and claims.
COPY data/raw/ ./data/raw/
COPY claims/ ./claims/
RUN mkdir -p data/external models reports && chown -R appuser:appuser /app

USER appuser
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["pwl"]
CMD ["--help"]
