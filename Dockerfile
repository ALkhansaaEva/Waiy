# Use official Python 3.9.13 base image
# Uncomment the next line if you want to use a smaller image with Alpine-based Python:
# FROM python:3.9.13-slim
# =====================================================================
# STAGE 1: The "Builder" Stage
# (This stage installs packages and stays the same)
# =====================================================================
FROM python:3.10-slim AS builder

# 1. Install OS Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Create a virtual environment
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV

# 3. Install Python Dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =====================================================================
# STAGE 2: The "Final" Stage
# (This is the final, optimized image)
# =====================================================================
FROM python:3.10-slim

# 1. Install OS Dependencies (We MUST add 'curl' here)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    curl \  # <-- إضافة أداة التحميل
    && rm -rf /var/lib/apt/lists/*

# 2. Set Working Directory
WORKDIR /app

# 3. Copy the virtual environment from the "builder" stage
COPY --from=builder /venv /venv

# 4. Copy Application Code
# (This copies app.py and the small 1KB "pointer" files)
COPY . .

# 5. (THE FIX) Download the REAL model file
# We use curl -L (follow redirects) to download the "raw" model
# and -o to overwrite the pointer file with the real one.
RUN curl -L \
    "https://github.com/ALkhansaaEva/Waiy/raw/main/model/enet_b2_7.pt" \
    -o "/app/model/enet_b2_7.pt"

# 6. Expose Port
EXPOSE 8000

# 7. Run Application
CMD ["/venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
