# Use a smaller base image (Debian-based slim variant)
FROM python:3.12-slim AS base

# Set working directory
WORKDIR /app

# Install dependencies required for pip packages (only minimal ones)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./

# Create virtual environment & install dependencies inside it
RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Use non-root user
RUN useradd -m appuser
USER appuser

# Set environment variables for optimization
ENV PATH="/venv/bin:$PATH" \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE="none" \
    STREAMLIT_GLOBAL_DEVELOPMENT_MODE="false"

# Run the Streamlit app
CMD ["streamlit", "run", "docbot.py"]
