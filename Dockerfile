# Use official Python 3.9.13 base image
# Uncomment the next line if you want to use a smaller image with Alpine-based Python:
# FROM python:3.9.13-slim
# 1. Base Image
# Using python 3.10-slim (Debian) which is compatible with PyTorch and OpenCV
FROM python:3.10-slim

# 2. Install OS Dependencies
# Install system libraries required by OpenCV (cv2) to run correctly
# libgl1/libglx-mesa0 are for graphics, libglib2.0-0 is a core library
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set Working Directory
WORKDIR /app

# 4. Copy requirements file
# Copy only requirements.txt first to leverage Docker's layer caching.
# This step won't re-run if only the app code changes.
COPY requirements.txt .

# 5. Install Python Dependencies
# Upgrade pip and install packages from requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --use-feature=fast-deps -r requirements.txt

# 6. Copy Application Code
# Copy the rest of the application files.
# This will now respect the .dockerignore file and skip large/unneeded files.
COPY . .

# 7. Expose Port
# Expose the port your FastAPI app runs on
EXPOSE 8000

# 8. Run Application
# Command to start the uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]