# SEKOSA: System Engineering Knowledge for Online Self-Assessment

## Anonymous Reproduction Package for ESWC 2026 Paper

This repository contains the complete implementation and demonstration materials for the SEKOSA approach presented in:

> **"A System Engineering Knowledge Ontology for Self-Assessment in a Robot"**  

---

## Overview

SEKOSA enables autonomous robots to assess their own capabilities under varying environmental conditions by:
- Modeling system components and their performance characteristics in an ontology
- Automatically inferring available configurations based on current conditions
- Estimating expected performance of different behaviors
- Selecting appropriate behaviors or recognizing inability to perform tasks

This package provides everything needed to reproduce the demonstration from the paper.

---

## Contents

```
.
├── README.md                           # This file
├── SEKOSA_schema.tql             		# Ontology definition (TypeDB 3.0.x)
├── SEKOSA_data.tql      				# Ontology instantiation and data used for the demonstration
├── sekosa_demonstration.py           	# Reproduces our demonstration
```

---

## Prerequisites

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

Using TypeDB requires certain Python versions.
We are using Python 3.11.14 with the packages listed below.

**Packages to install:**
- typedb-driver == 2.28.0
- matplotlib >= 3.10.7
- numpy >= 2.3.5
- pandas >= 2.3.3
---

## Quick Start

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
source SEKOSA_demo_data.tql

# Commit the sourced data
commit
```

### Step 3: Run the Demonstration
**Option B: Standalone Python Script**

In another terminal, run:
```bash
python3.11 SEKOSA_demonstration.py
```

