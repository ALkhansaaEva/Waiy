# Use official Python 3.9.13 base image
# Uncomment the next line if you want to use a smaller image with Alpine-based Python:
# FROM python:3.9.13-slim
# 1. Use 'slim' (Debian-based) instead of 'alpine'
# This is required for complex binaries like torch, torchvision, and opencv-python.
FROM python:3.11-slim

# 2. Install OS dependencies for OpenCV (libgl1)
# This prevents errors like "libGL.so.1: cannot open shared object file"
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 3. Install Python packages
# I removed the experimental '--use-feature=fast-deps' for better stability
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
# This copies app.py, the 'model' folder, and 'static' folder
COPY . .

# Expose the port and run the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]