# Privacy Risk Assessment on Reddit through Data-Driven Auditing and Visualization

This repository accompanies the MSc dissertation *Privacy Risk Assessment on Reddit through Data-Driven Auditing and Visualization* (University of Reading, 2025).  
The project implements a reproducible pipeline to audit privacy risks on Reddit using public data, weak supervision, and interpretable machine learning.

---

## Features
- Reddit data collection via the official API (`praw`)  
- Feature engineering for six privacy risk dimensions: Identity, Sensitive Content, Exposure, Activity, Volume, Concentration  
- Weak supervision rules for heuristic risk labels  
- LightGBM baseline model with calibration  
- Cross-validation and evaluation metrics (F1, AUROC, PR-AUC)  
- Risk scoring and visualization via Streamlit  

---

## Installation

### 1. Clone the repository
```bash
git clone https://gitlab.act.reading.ac.uk/ep839056/privacy_audit_redditmsc-project.git
cd privacy_audit_redditmsc-project
```

### 2. Create environment (Python 3.11+ recommended)
Using Conda:
```bash
conda create -n reddit_privacy python=3.11
conda activate reddit_privacy
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Or with Poetry (if using `pyproject.toml`):
```bash
poetry install
```

---

## Usage

### Data collection
Fetch Reddit users and posts (requires Reddit API credentials in `.env` or `config.py`):
```bash
python src/app.py crawl-users --data data/raw/users.json
```

### Feature extraction
```bash
python src/app.py featurize --data data/raw/users.json --out data/processed/features.csv
```

### Weak supervision labelling
```bash
python src/app.py auto-label --data data/processed/features.csv --out data/processed/train.csv
```

### Model training with cross-validation
```bash
python src/app.py train-cv --data data/processed/train.csv --outdir models/cv --folds 5
```

### Launch Streamlit dashboard
```bash
streamlit run src/dashboard.py
```

---

## Ethics
- Only public Reddit data was used.  
- No identity inference or cross-platform linkage was attempted.  
- All analysis is aggregate; usernames were anonymised after feature extraction.  

---

## Citation
If you use this repository, please cite:

> Zheng, E. (2025). *Privacy Risk Assessment on Reddit through Data-Driven Auditing and Visualization*. MSc Dissertation, University of Reading.
