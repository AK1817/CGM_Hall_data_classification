# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app


# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python code
COPY Hall.py .
COPY Hall ./Hall

# Run the script
CMD ["python", "Hall.py"]