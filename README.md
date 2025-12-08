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
## Instructions
1. Install TypeDB
2. Create a TypeDB database with the ontology as schema and the instantiations as data.
3. Run the python script to repoduce our results.

Commandline reproduction comming soon...

