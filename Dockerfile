FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --requirement requirements.txt

# Download model assets during the image build to avoid network-dependent
# downloads and longer startup times when Cloud Run creates a new instance.
RUN mkdir -p "$NLTK_DATA" "$HF_HOME" \
    && python -c "import nltk; nltk.download('wordnet', download_dir='/app/nltk_data'); nltk.download('omw-1.4', download_dir='/app/nltk_data')" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["python", "src/dialog_bot/app.py"]
