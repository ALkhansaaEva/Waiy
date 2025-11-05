# Use official Python 3.9.13 base image
# Uncomment the next line if you want to use a smaller image with Alpine-based Python:
# FROM python:3.9.13-slim
# 1. Use 'slim' (Debian-based) as the base
FROM python:3.10-slim

# 2. Install correct OS dependencies for OpenCV
#    - 'libgl1' and 'libglx-mesa0' are the modern replacements for 'libgl1-mesa-glx'
#    - 'libglib2.0-0' is still needed
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# 3. Install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY . .

# Expose the port and run the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]