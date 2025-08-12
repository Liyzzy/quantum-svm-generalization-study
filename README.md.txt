# Generalization Comparison Study Between Quantum SVM and Linear SVM
This project investigates the performance and generalization capabilities of Quantum Support Vector Machines (QSVM) versus Classical SVM in binary classification tasks. Implemented using Qiskit on simulated quantum backends (Qiskit Aer) and compared with traditional SVMs in scikit-learn, the study evaluates two datasets Spambase and Lung Cancer.

## Overview
- **Objective:** Compare QSVM and classical SVM in terms of accuracy, precision, recall, F1-score, and generalization gap.
- **Datasets:**
  - *Spambase*
  - *Lung Cancer*
- **Methods:**
  - Preprocessing and PCA for dimensionality reduction.
  - Classical SVM with linear kernel.
  - QSVM using ZZFeatureMap and quantum kernel estimation.
  - Noise-aware simulations using Qiskit Aer.

## Features
- QSVM implementation with quantum kernel estimation.
- Classical SVM baseline.
- Noise simulation for NISQ conditions.
- Detailed evaluation metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Generalization gap
- Insights from quantum information theory on model generalization.

## Frameworks Used
- Python 3.x
- Qiskit
- scikit-learn
- NumPy, Pandas, Matplotlib

## 🚀 Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/Lizzy/quantum-svm-generalization-study.git
   cd quantum-svm-generalization-study

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
4. Run
   ```bash
   python src/run_experiments.py