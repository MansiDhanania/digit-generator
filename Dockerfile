# Multi-stage build for minimal image size
FROM python:3.11-slim as builder

WORKDIR /tmp

# Copy only production requirements
COPY requirements-prod.txt requirements.txt ./

# Install build dependencies and create slim requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --user --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --user --no-cache-dir -r requirements-prod.txt \
    && apt-get remove -y build-essential && apt-get autoremove -y

# Final stage - minimal runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only the compiled Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Copy application files only
COPY app.py .
COPY config.py .
COPY cvae_mnist.py .
COPY cvae_mnist.pth .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
