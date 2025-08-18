# SMC Trading Bot - Web Interface Dockerfile
FROM python:3.11-slim

LABEL maintainer="SMC Trading Bot"
LABEL description="Smart Money Concepts Trading Bot - Web Interface"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY web/backend/ ./web/backend/
COPY web/frontend/ ./web/frontend/
COPY config.py .
COPY dot.env.example .env

# Create necessary directories
RUN mkdir -p data signals_history

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Start command
CMD ["python", "start_web_panel.py"]
