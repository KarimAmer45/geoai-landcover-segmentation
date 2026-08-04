FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
COPY data ./data
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

CMD ["pytest", "-m", "not network", "-q"]

