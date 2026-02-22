# 1. Base Image
FROM python:3.11-slim

# 2. Linux Updates
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Copy Requirements
COPY requirements.txt .

# 5. Install Everything in one go (CPU Optimized)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the app
COPY . .

# 7. Port Expose 
EXPOSE 7860

# 8. Start App
CMD ["python", "app.py"]