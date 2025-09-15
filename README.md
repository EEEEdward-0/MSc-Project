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

## Version History

- **v1.0.0 (Sep 2025)**  
  Initial release with full pipeline (data collection, featurization, weak supervision, model training, visualization).

- **v1.1.0 (Sep 2025)**  
  Added Quickstart demo mode with synthetic data, improved Streamlit dashboard compatibility, and reproducibility pack.

- **v1.2.0 (Demo branch, Sep 2025)**  
  Dedicated branch for lightweight demo.  
  Runs with bundled synthetic data only, skips Reddit API.  
  Simplified training/evaluation for presentation purposes.

- **v1.2.1 (Demo branch, Sep 2025)**  
  Refined demo branch with added feature toggles for local training vs demo mode, improved documentation, and synchronization support for GitHub and GitLab.

---

## Changelog

### v1.2.1 (Sep 2025)
- Added toggle in Streamlit UI to switch between Demo mode and Local (full) mode.
- Improved documentation with Demo vs Full Workflow comparison table.
- Synced repository across GitHub and GitLab.

### v1.2.0 (Sep 2025)
- Created dedicated Demo branch with bundled synthetic data.
- Simplified training and evaluation flow for presentation.
- Removed direct Reddit API dependency in demo branch.

### v1.1.0 (Sep 2025)
- Added Quickstart demo script and reproducibility pack.
- Enhanced Streamlit dashboard compatibility.

### v1.0.0 (Sep 2025)
- Initial release with complete pipeline:
  - Data collection
  - Feature engineering
  - Weak supervision
  - Model training & evaluation
  - Streamlit visualization

---

## Minimal Demo (Recommended for quick testing)

You can run the demo without Reddit API credentials or raw data. This is the simplest way for supervisors/examiners to verify the project.

1. **Clone repository**
   ```bash
   git clone https://github.com/EEEEdward-0/MSc-Project.git
   cd MSc-Project
   ```

2. **Set up environment**
   ```bash
   conda env create -f reproducibility/environment.yml
   conda activate reddit_privacy
   ```
   or
   ```bash
   pip install -r reproducibility/requirements.txt
   ```

3. **Run demo script**
   ```bash
   bash scripts/quickstart_demo.sh
   ```

   This will:
   - Verify dependencies
   - Generate synthetic demo features and models
   - Launch the Streamlit dashboard

4. **Open dashboard**
   Visit [http://localhost:8501](http://localhost:8501) in your browser.

   In the sidebar, you will see options for:
   - **Auto Detect / Local (trained)**
   - **Demo only (sliders)** ← active in demo mode

---

## Demo vs Full Workflow (at a glance)

| Aspect             | Demo Mode (default)                        | Full Workflow (research)               |
|--------------------|--------------------------------------------|----------------------------------------|
| Data source        | Synthetic demo data (bundled)              | Reddit API + raw user/post data (local)|
| Featurization      | Pre-generated / synthetic features         | Full feature extraction from raw data  |
| Labelling          | Fixed heuristic demo labels                | Weak supervision (auto-label)          |
| Models             | Pretrained lightweight demo models         | Cross-validated training (LogReg / LGBM / RF) |
| Reproducibility    | Runs instantly, no external dependencies   | Requires API credentials & raw dataset |
| Streamlit dashboard| **Demo only** sliders active               | Full functionality with training/eval  |

---

## Full Workflow (Research mode)

**Warning:** Reddit API credentials and raw data are required for the full workflow and are not included in this repository.

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
## Reproducibility Pack

A minimal reproducibility pack is provided under the `reproducibility/` folder.  
It contains environment specifications, example configuration, and a quickstart script.

### Contents
- `environment.yml` – Conda environment definition  
- `requirements.txt` – Python package requirements (pip)  
- `.env.example` – Example Reddit API credentials file (copy to `.env`)  
- `run_quickstart.sh` – One-click script to verify dependencies and create a synthetic dataset  
- `scripts/verify_setup.py` – Python script to verify environment and data  
- `data/README.txt` – Explanation of data folders  

### Quickstart
1. Create the environment:
```bash
conda env create -f reproducibility/environment.yml
conda activate reddit_privacy
```
 ----

## Ethics
- Only public Reddit data was used.  
- No identity inference or cross-platform linkage was attempted.  
- All analysis is aggregate; usernames were anonymised after feature extraction.  

## References

- Nissenbaum, H. (2004). Privacy as contextual integrity. Washington Law Review, 79(1), 119–158.  
- Solove, D. J. (2006). A taxonomy of privacy. University of Pennsylvania Law Review, 154(3), 477–564.  
- Narayanan, A., & Shmatikov, V. (2008). Robust de-anonymization of large sparse datasets. IEEE Symposium on Security and Privacy.  
- Ratner, A., et al. (2017). Snorkel: Rapid training data creation with weak supervision. PVLDB, 11(3), 269–282.  
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 30.  

## Citation
If you use this repository, please cite:

> Zheng, E. (2025). *Privacy Risk Assessment on Reddit through Data-Driven Auditing and Visualization*. MSc Dissertation, University of Reading.
