This repository contains the full implementation and analysis for the project titled:

**Cross-Domain Fake News Detection with Interpretable Linguistic Features on Reddit and ISOT** by Harun Eroglu  

## Project Overview

This project investigates the use of interpretable linguistic, syntactic, and structural features to detect fake news across two different domains: Reddit (social media) and ISOT (formal news). Traditional machine learning models are employed to emphasize transparency and reproducibility.

---

## Files and Purpose

| File | Purpose |
|------|---------|
| `1_api_scraper_reddit.py` | Collect Reddit posts from specified subreddits using Reddit API. |
| `1_reddit_dataset_uncleaned.csv` | Raw Reddit dataset scraped using the script above. |
| `2_reddit_dataset_preprocess.py` | Clean Reddit text and extract all linguistic, syntactic, and structural features. |
| `3_isot_fake_uncleaned.csv`, `3_isot_real_uncleaned.csv` | External ISOT dataset files (not included due to GitHub file size limits). |
| `4_isot_dataset_preprocess.py` | Clean ISOT news articles and extract the same features as Reddit. |
| `5_logistic_regression_train.py` | Train and evaluate Logistic Regression on Reddit and ISOT datasets. |
| `5_rf_xgb_train.py` | Train and evaluate Random Forest and XGBoost on both datasets. |
| `6_feature_importance_results.py` | Compute and visualize feature importances for each model-dataset pair. |

---

## Dataset Notice

Due to GitHub’s 100 MB file size limit, the ISOT dataset (`True.csv`, `Fake.csv`) is **not included** in this repository.  
You can download it from Kaggle:

https://www.kaggle.com/datasets/snapcrack/all-the-news

> Only the first two files (`True.csv` and `Fake.csv`) are used and their names were changed as "3_isot_fake_uncleaned" and "3_isot_true_uncleaned".

---

## How to Run

### Setup

```bash
python3 -m venv cleanenv
source cleanenv/bin/activate
pip install -r requirements.txt
```

### Step-by-step Execution

```bash
# 1. Scrape Reddit dataset (optional since CSV already exists)
python 1_api_scraper_reddit.py

# 2. Preprocess Reddit and ISOT datasets
python 2_reddit_dataset_preprocess.py
python 4_isot_dataset_preprocess.py

# 3. Train models
python 5_logistic_regression_train.py
python 5_rf_xgb_train.py

# 4. Feature importance
python 6_feature_importance_results.py
```

---

## Output Files

- `2_reddit_dataset_preprocessed.csv`
- `4_isot_dataset_preprocessed.csv`
- `5_logistic_regression_results.csv`
- `5_rf_xgb_results.csv`
- `6_feature_importance_combined.csv`
- `6_feature_importance_combined_heatmap.png`

---

## License

This project is open for academic and research purposes. Please cite appropriately if used.

## Contact

For any questions or collaborations, feel free to contact.
