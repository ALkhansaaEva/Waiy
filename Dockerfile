# Use official Python 3.9.13 base image
# Uncomment the next line if you want to use a smaller image with Alpine-based Python:
# FROM python:3.9.13-slim
# =====================================================================
# STAGE 1: The "Builder" Stage
# This stage installs all Python packages
# It will be cached as long as requirements.txt doesn't change
# =====================================================================
FROM python:3.12-slim AS builder

# 1. Install OS Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Create a virtual environment
# We install packages into /venv instead of the global site-packages
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV

# 3. Install Python Dependencies
# Copy only requirements.txt and install them into the venv
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================================================================
# STAGE 2: The "Final" Stage
# This is the final, optimized image for production
# =====================================================================
FROM python:3.12-slim

# 1. Install OS Dependencies (must be in final image too)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Working Directory
WORKDIR /app

# 3. Copy the virtual environment from the "builder" stage
# This is the magic! We copy the pre-installed packages (1.5 min step)
COPY --from=builder /venv /venv

# 4. Copy Application Code
# This respects .dockerignore and is very fast
COPY . .

# 5. Expose Port
EXPOSE 8000

# 6. Run Application
# We use the python from the venv to run the app
CMD ["/venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]