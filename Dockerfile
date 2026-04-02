FROM python:3.11-slim

# Install system dependencies for GDAL/rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create outputs directory
RUN mkdir -p /app/outputs

EXPOSE 8000

# Use SERVICE_TYPE env var to determine what to run (default: scheduler)
# Set SERVICE_TYPE=web for the Streamlit dashboard
CMD ["sh", "-c", "if [ \"$SERVICE_TYPE\" = 'web' ]; then streamlit run ui/app.py --server.port ${PORT:-8080} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false; else python scheduler_service.py; fi"]
