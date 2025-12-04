# SEKOSA: System Engineering Knowledge for Online Self-Assessment

## Anonymous Reproduction Package for Review

This repository contains the complete implementation and demonstration materials for our paper presenting the SEKOSA approach.

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
├── SEKOSA_ontology.tql             # Ontology definition
├── SEKOSA_demo_instantiation.tql      # Ontology instantiation and data used for the demonstration
├── sekosa_demonstration.py           # Python script for reproducing demonstration
```

---

## Prerequisites

### 1. TypeDB 3.0.x

Download and install TypeDB from: https://typedb.com/download

**Quick installation:**
```bash
# macOS (Homebrew)
brew install typedb

# Linux (manual)
wget https://github.com/typedb/typedb/releases/download/3.0.0/typedb-all-linux-x86_64-3.0.0.tar.gz
tar -xzf typedb-all-linux-x86_64-3.0.0.tar.gz
cd typedb-all-linux-x86_64-3.0.0

# Start TypeDB server
./typedb server
```

TypeDB will run on `localhost:1729` by default.

### 2. Python Requirements

```bash
# Create virtual environment (recommended)
python3 -m venv sekosa-env
source sekosa-env/bin/activate  # On Windows: sekosa-env\Scripts\activate

# Install dependencies
pip install typedb-driver matplotlib numpy pandas jupyter
```

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
# Create the database
typedb console --command="database create sekosa"

# Load the schema (ontology)
typedb console --database=sekosa --file=SEKOSA_ontology_v3.tql

# Load the data (component instantiation)
typedb console --database=sekosa --file=SEKOSA_knowledge-base_v3.tql
```

### Step 3: Run the Demonstration
**Option B: Standalone Python Script**
```bash
python sekosa_demo_script.py
```

