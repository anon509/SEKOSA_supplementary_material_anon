# SEKOSA: System Engineering Knowledge for Online Self-Assessment

## Anonymous Reproduction Package for J-WOSMARS 2026 Paper

This repository contains the complete implementation and demonstration materials for the SEKOSA approach presented in:

> **"A System Engineering Knowledge Ontology for Self-Assessment in a Robot"**  

---

## Overview

SEKOSA enables autonomous robots to assess their own capabilities under varying environmental conditions by:
- Modeling system components and their performance characteristics in an ontology
- Automatically inferring available configurations based on current conditions
- Estimating expected performance of different behaviors
- Selecting appropriate behaviors or recognizing inability to perform tasks

This package provides resources and instructions needed to reproduce the demonstration from the paper.

---

## Contents

```
.
├── README.md                           # This file
├── SEKOSA_schema.tql             		# Ontology definition
├── SEKOSA_data.tql      				# Component and requirement instantiation
├── SEKOSA_demonstration.py           	# Reproduces our demonstration
├── docker/                             # Docker setup files
    ├── Dockerfile
    ├── docker-compose.yml
    ├── docker-entrypoint.sh
    ├── initialize_database.py
```

---

## Quick Start with Docker (Recommended)

The easiest way to reproduce the demonstration is using Docker. This automatically sets up TypeDB, Python, and all dependencies.

### Requirements
- Docker installed (check with `docker --version`)
- ~2GB RAM available
- ~1GB disk space

### Steps

```bash
# 1. Navigate to the docker directory
cd docker

# 2. Build and start the container (first time takes a few minutes)
sudo docker compose up -d

# 3. Run the demonstration
sudo docker exec -it sekosa-container /opt/venv/bin/python /app/SEKOSA_demonstration.py

# 4. View results
# The plot is saved as: sekosa_demonstration_results.png in the project root

# 5. Stop the container when done
sudo docker compose down
```

**Note:** If you get permission errors, add your user to the docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker
```
Then you can run without `sudo`.

---

## Manual Setup

If you prefer to run without Docker, follow these prerequisites:

### 1. TypeDB 2.28.3

**Installation:**

Install JDK

```bash
sudo apt install default-jdk
```

Install TypeDB (instructions are for Ubuntu)
```bash
sudo apt install software-properties-common apt-transport-https gpg 

gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 17507562824cfdcc 

gpg --export 17507562824cfdcc | sudo tee /etc/apt/trusted.gpg.d/typedb.gpg > /dev/null 

echo "deb https://repo.typedb.com/public/public-release/deb/ubuntu trusty main" | sudo tee /etc/apt/sources.list.d/typedb.list > /dev/null 

sudo apt update 

sudo apt install typedb=2.28.3
```

### 2. Python Requirements

Using TypeDB 2.28.3 requires certain Python versions.
We are using Python 3.11.14 with the packages listed below.

**Packages to install:**
- typedb-driver == 2.28.0
- matplotlib >= 3.10.7
- numpy >= 2.3.5
- pandas >= 2.3.3

---

## Manual Quick Start

### Step 1: Start TypeDB Server

```bash
typedb server
```

Leave this running in a terminal.

### Step 2: Load the Ontology and Data

In a new terminal:

```bash
# Start the console
sudo typedb console

# Create a TypeDB database
database create sekosa

# Enter the schema transaction
transaction sekosa schema write

# Source the SEKOSA schema
source SEKOSA_schema.tql

# Commit the sourced schema
commit

# Enter the data transaction
transaction sekosa data write

# Source the SEKOSA data
source SEKOSA_data.tql

# Commit the sourced data
commit
```

### Step 3: Run the Demonstration

In another terminal, run:
```bash
python SEKOSA_demonstration.py
```

