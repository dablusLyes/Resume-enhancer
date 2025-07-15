# Use official Python image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for pdf2image and Pillow
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y poppler-utils gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy French model at build time
RUN python -m spacy download fr_core_news_sm

# Copy backend code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the Flask server
CMD ["python", "server.py"]