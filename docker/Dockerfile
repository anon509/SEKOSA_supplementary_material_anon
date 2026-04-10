# SEKOSA Docker Container
# Includes TypeDB 2.28.3 and Python environment for running the demonstration

FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    default-jdk \
    software-properties-common \
    apt-transport-https \
    gpg \
    wget \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install TypeDB 2.28.3
RUN gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 17507562824cfdcc && \
    gpg --export 17507562824cfdcc > /etc/apt/trusted.gpg.d/typedb.gpg && \
    echo "deb https://repo.typedb.com/public/public-release/deb/ubuntu trusty main" > /etc/apt/sources.list.d/typedb.list && \
    apt-get update && \
    apt-get install -y typedb=2.28.3 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files (context is parent directory)
COPY SEKOSA_schema.tql .
COPY SEKOSA_data.tql .
COPY SEKOSA_demonstration.py .
COPY README.md .

# Create Python virtual environment and install dependencies
RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install \
    typedb-driver==2.28.0 \
    matplotlib>=3.10.7 \
    numpy>=2.3.5 \
    pandas>=2.3.3

# Copy initialization script
COPY docker/docker-entrypoint.sh /docker-entrypoint.sh
COPY docker/initialize_database.py /app/initialize_database.py
RUN chmod +x /docker-entrypoint.sh

# Expose TypeDB port
EXPOSE 1729

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
