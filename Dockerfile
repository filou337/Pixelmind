FROM python:3.12-slim

WORKDIR /app
# CPU-only torch: lets the API serve deep-learning checkpoints without CUDA libraries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml,rag,agents,ui]"
COPY app ./app
COPY docs ./docs

# run as an unprivileged user; artifacts/ is a mounted volume
RUN useradd --create-home --uid 1000 pixelmind
USER pixelmind

ENV PIXELMIND_ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"
CMD ["uvicorn", "pixelmind.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
