# Dockerfile
FROM python:3.11-slim

# Avoid writing .pyc, and make logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but nice): certificates + curl for debugging
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Python deps
# - polars: dataframe
# - pyarrow: reliable parquet read/write backend
# - requests: http
# - beautifulsoup4: html->plain text
RUN pip install --no-cache-dir \
    polars==1.* \
    pyarrow==17.* \
    requests==2.* \
    beautifulsoup4==4.*

# Copy your script into the image
RUN git clone https://github.com/fulaibaowang/dictycite.git
COPY get_curated_notes.py /app/get_curated_notes.py

# # Default command (you can override at `docker run` time)
# CMD ["python", "/app/get_curated_notes.py", "--limit", "10"]
