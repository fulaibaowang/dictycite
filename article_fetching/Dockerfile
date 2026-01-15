# Dockerfile for running article_fetching/fetch.py
# Base as requested
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Install system deps for Python packages and R (for tidypmc via rpy2)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       python3-dev \
       wget \
       git \
       ca-certificates \
       libxml2-dev \
       libssl-dev \
       libcurl4-openssl-dev \
       libffi-dev \
       gfortran \
       pkg-config \
       r-base \
       r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python dependencies used by fetch.py and utils
RUN pip install --no-cache-dir requests tqdm pandas rpy2

# Install common R packages used by utils/get_text/epmc_xml.py
# (tidypmc depends on XML/httr/etc.; install commonly required packages first)
RUN Rscript -e "install.packages(c('xml2','httr','jsonlite','dplyr'), repos='https://cloud.r-project.org')"
# Try to install tidypmc from CRAN; if not available CRAN, user may install manually
RUN Rscript -e "if(!requireNamespace('tidypmc', quietly=TRUE)) install.packages('tidypmc', repos='https://cloud.r-project.org')"

# Copy your script into the image
RUN git clone https://github.com/fulaibaowang/dictycite.git

WORKDIR /dictycite/article_fetching

# Default entrypoint runs fetch.py; pass args through docker run
ENTRYPOINT ["python", "fetch.py"]
CMD ["--help"]
