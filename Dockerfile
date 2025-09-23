# Use official Python 3.9.13 base image
FROM python:3.9.13-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
