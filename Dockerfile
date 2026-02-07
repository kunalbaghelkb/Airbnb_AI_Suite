# 1. Base Image
FROM python:3.11-slim

# 2. Linux Updates & Cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Copy Requirements
COPY requirements.txt .

# 5. OPTIMIZATION: Install CPU-only PyTorch FIRST
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install Requirements
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 7. Copy App
COPY . /app

# 8. Port Expose
EXPOSE 8000

# 9. Start App
CMD ["python", "app.py"]