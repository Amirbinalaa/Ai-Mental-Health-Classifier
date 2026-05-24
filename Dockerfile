FROM python:3.11-slim

ENV HF_HOME=/root/.cache/huggingface
ENV PORT=7860
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure necessary NLTK data is downloaded during build
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"

# Create temp directory
RUN mkdir -p src/data/temp

# Copy project files
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# Expose the Hugging Face default port
EXPOSE 7860

# Start both services
CMD ["./start.sh"]
