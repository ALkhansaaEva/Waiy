# Use official Python 3.9.13 base image
# Uncomment the next line if you want to use a smaller image with Alpine-based Python:
# FROM python:3.9.13-slim
# Using Python 3.11-alpine image for lightweight setup, but note that some packages may not be compatible with Alpine.
FROM python:3.11-alpine

# Set working directory inside the container
WORKDIR /app

# Copy dependency file (requirements.txt) into the container
COPY requirements.txt .

# Install pip dependencies using the 'no-cache' option to avoid caching of package downloads
# Also using the 'fast-deps' feature which is experimental, it can improve install speed but might cause issues in production.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --use-feature=fast-deps -r requirements.txt

# Copy all project files into the container, including code, configuration, etc.
COPY . .

# Expose port 8000 for the FastAPI application to be accessible
EXPOSE 8000

# Run the FastAPI app using uvicorn as the ASGI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
