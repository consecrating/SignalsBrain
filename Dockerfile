FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir fastapi uvicorn httpx pydantic numpy scipy pyyaml python-jose

# Copy source
COPY . .

# Create data directory
RUN mkdir -p data

EXPOSE 8400

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8400"]
