FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV, OCR, and basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    file \
    # OpenCV dependencies for Debian 12
    libglx-mesa0 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libstdc++6 \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    # Additional utilities
    wget \
    curl \
    git \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads data/vector_store data/backups logs

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860
ENV OLLAMA_ENABLED=false
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app:$PYTHONPATH
ENV OMP_NUM_THREADS=1
ENV PADDLE_PADDLE_NO_WARN=1
ENV PADDLE_NO_WARN=1

# Expose port for HF Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/api/health', timeout=5)" || exit 1

# Start application with optimized settings for Hugging Face Spaces
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "120"]