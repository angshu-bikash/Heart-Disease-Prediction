# Heart Failure Prediction using Calibrated Ensemble Learning

This repository supports the Master's Thesis (PRT840 – IT Thesis) submitted to **Charles Darwin University (2025)**.  
It presents a reproducible framework for **heart-disease prediction** using both **classical machine-learning** and **deep-learning** models on a **unique, harmonised dataset** combining **five independent repositories** (Cleveland, Hungary, Switzerland, Long Beach, and Statlog).  
The final integrated dataset contains **918 patient records** and **12 clinical attributes**.

## Overview
The project implements an end-to-end experimental pipeline covering:
- Data integration and preprocessing  
- Model development (Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, SVM, MLP, LSTM, GRU)  
- Hyperparameter tuning and isotonic calibration  
- Ensemble integration (SVM + MLP)  
- Fairness and explainability analysis (SHAP, permutation importance)  

## Experimental Environment
| Component | Version / Tool |
|------------|----------------|
| Python | 3.10 |
| TensorFlow / Keras | 2.15 |
| scikit-learn | 1.5 |
| pandas / NumPy | 2.1 / 1.26 |
| matplotlib / seaborn | 3.9 / 0.13 |
| SHAP | 0.45 |
| Hardware | Intel i7 CPU · 16 GB RAM · NVIDIA RTX 3060 GPU |

> All experiments used deterministic seeding (42) for reproducibility.  
> Dependencies are listed in `requirements.txt`.

## Parameter Settings
- Cross-validation: 5-fold stratified  
- Optimiser: Adam (lr = 0.001)  
- Neural batch size = 32, epochs ≤ 100, dropout = 0.2  
- Ensemble weighting: α = 0.60 for SVM, 0.40 for MLP  
- Calibration: Isotonic regression (preferred) and Platt scaling (comparative)

## Repository Structure
heart-failure-prediction-thesis/
├── data/ # metadata or synthetic samples only
├── notebooks/ # Final_Model.ipynb
├── results/ # ROC, calibration, SHAP plots
├── requirements.txt
├── README.md
└── LICENSE
