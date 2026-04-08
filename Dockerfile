FROM python:3.12-slim AS builder

WORKDIR /app

# Copy everything needed for install
COPY pyproject.toml README.md ./
COPY app/ app/
COPY providers/ providers/
COPY context_engine/ context_engine/
COPY db/ db/
COPY worker/ worker/

RUN pip install --no-cache-dir .

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

RUN useradd -r -s /bin/false appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
