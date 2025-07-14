FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 cognitive
USER cognitive

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from reservoirpy.cognitive import UnifiedCognitiveSystem; UnifiedCognitiveSystem()"

# Start application
CMD ["python", "-m", "reservoirpy.cognitive.server"]
