# CSE-4705-Team-3

# Project Directory
- data/               # Raw and preprocessed note images
- notebooks/          # Exploratory analysis and quick tests
- src/                # Shared helper functions (e.g., resizing, binarizing)
- models/             # Models
- results/            # Final comparison plots and tables
- requirements.txt    # List of libraries

# Installation and Environment Setup

This project uses **Python 3.12**. Installation instructions are written in terms of Linux commands.

### 1. Prerequisites
Ensure you have **Python 3.12** installed on your system.
* Check your version: `python --version` or `python3 --version`.
* **Note:** Using Python 3.13+ may cause TensorFlow installation failures.

### 2. Repository Setup
Clone the repository and navigate into the project folder:
```bash
git clone git@github.com:AbhiSwamiUConn/CSE-4705-Team-3.git
cd CSE-4705-Team-3
```

### 3. Virtual Environment
Use a virtual environment to isolate project dependencies:
```bash
python3.12 -m venv venv
source venv/bin/activate
```
Install dependencies via pip:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
To exit the venv at any time:
```bash
deactivate
```
