# MALLORN Astronomical Classification Challenge

A machine learning competition repository for the **MALLORN Astronomical Classification Challenge**, focused on **detecting rare Tidal Disruption Events (TDEs)** from simulated **LSST lightcurve time-series data**. The challenge is highly imbalanced and requires robust feature engineering, probabilistic modeling, and threshold optimization. ([kaggle.com](https://www.kaggle.com/datasets/barkataliarbab/mallorn-dataset?utm_source=chatgpt.com))

---

## Project Overview

This project builds an end-to-end **astronomical time-series classification pipeline** to identify rare TDE events from multi-band photometric observations.

The MALLORN dataset contains over **10,000 simulated LSST lightcurves**, generated from real ZTF observations of:

* Tidal Disruption Events (TDEs)
* Active Galactic Nuclei (AGN)
* Nuclear Supernovae

The main challenge is **extreme class imbalance**, where TDEs are rare compared with other transient events. ([arxiv.org](https://arxiv.org/abs/2512.04946?utm_source=chatgpt.com))

---

## Repository Structure

```bash
MALLORN-Astronomical-Classification-Challenge/
│
├── notebooks/               # Main experiments and EDA
├── src/                     # Feature engineering and modeling pipeline
├── outputs/                 # Submission files and prediction results
├── figures/                 # Lightcurve and feature plots
├── submission.csv           # Final Kaggle submission
└── README.md                # Project documentation
```

> Adjust this structure to match your actual repository files.

---

## Problem Statement

The objective is to classify whether an astronomical transient is a **Tidal Disruption Event (TDE)** using only photometric time-series observations.

Key challenges:

* irregular observation cadence
* noisy flux measurements
* multiple photometric bands
* long temporal horizon
* severe target imbalance
* rare positive events

This makes the problem a combination of:

* **time-series classification**
* **astronomical feature extraction**
* **imbalanced binary classification**
* **probability calibration** ([kaggle.com](https://www.kaggle.com/datasets/barkataliarbab/mallorn-dataset?utm_source=chatgpt.com))

---

## Methodology

### 1) Data Preprocessing

* sort observations by object and timestamp
* normalize relative observation time
* missing value imputation
* invalid flux cleaning
* band-wise aggregation
* noise-aware weighting using flux error

### 2) Feature Engineering

Feature extraction is the most critical stage.

Possible features used:

* statistical flux summaries
* rise and decay slope
* peak brightness
* color differences between bands
* time-to-peak
* skewness and kurtosis
* variability index
* Bazin lightcurve fit parameters
* Gaussian Process trend features

These features help convert sparse astronomical sequences into model-ready tabular inputs.

---

### 3) Modeling

Recommended competition-grade models:

* LightGBM
* XGBoost
* CatBoost
* Ensemble Boosting
* Probability Averaging Ensemble

For sequential deep learning extensions:

* LSTM
* GRU
* Temporal CNN

---

### 4) Imbalance Handling

Because TDEs are rare, imbalance strategy is crucial.

Methods used:

* class weights
* focal threshold tuning
* stratified cross validation
* probability calibration
* percentile threshold optimization

In astronomy tasks, **threshold tuning often performs better than naive oversampling**.

---

## Workflow

```text
Load lightcurve data
        ↓
Preprocess temporal observations
        ↓
Band-wise feature extraction
        ↓
Time-series statistical summaries
        ↓
Model training
        ↓
Threshold optimization
        ↓
Submission generation
```

---

## Tech Stack

* Python
* pandas
* NumPy
* LightGBM
* XGBoost
* CatBoost
* SciPy
* scikit-learn
* matplotlib
* astroML / tsfresh (optional)

---

## Results

> Replace this section with your actual leaderboard score.

```text
Best Validation F1     : 0.66
Best Public Score      : Top 10%
Final Strategy         : Ensemble + Threshold Tuning
```

Suggested additions:

* confusion matrix
* PR curve
* threshold sensitivity analysis
* feature importance plot

---

## Key Insights

Important findings usually seen in this challenge:

* **relative time normalization improves convergence**
* multi-band color features are highly predictive
* ensemble boosting is strong for tabularized lightcurves
* threshold tuning significantly improves F1
* class weighting is safer than SMOTE for physical data
* GP/Bazin fit features capture astrophysical signal shape well

These insights align strongly with the official MALLORN challenge motivation. ([arxiv.org](https://arxiv.org/abs/2512.04946?utm_source=chatgpt.com))

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

For script pipeline:

```bash
python train.py
```

---

## Future Improvements

* sequence-based GRU modeling
* contrastive pretraining on lightcurves
* physics-informed feature extraction
* automated threshold search
* Optuna hyperparameter tuning
* stacking ensemble
* uncertainty-aware predictions

---

## Competition Reference

Kaggle Competition:
**MALLORN Astronomical Classification Challenge** ([kaggle.com](https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/overview/citation?utm_source=chatgpt.com))

---

## Author

**Dimas Pasha Akrilian**

This repository is part of my **astronomical machine learning and Kaggle competition portfolio**, specializing in **time-series classification, rare event detection, and scientific ML workflows**.
