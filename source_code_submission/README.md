# Quantum SVM Generalization Study — Source Code

This folder contains the source code developed for the research project.

## Folder Structure

| Folder | Description |
|--------|-------------|
| `01_classical_svm/` | Classical SVM experiments (PCA & SelectKBest feature selection) for Spambase and Lung Cancer datasets |
| `02_ideal_qsvm/` | Ideal (noiseless) Quantum SVM experiments using SVC-precomputed kernel methods |
| `03_noisy_qsvm/` | Noisy Quantum SVM experiments with simulated quantum noise |
| `04_error_mitigated_qsvm/` | Error-mitigated QSVM experiments using Zero-Noise Extrapolation and Readout Error Mitigation |
| `05_real_hardware/` | ONLY TEST - Experiments run on real quantum hardware (IBM Q, Microsoft Azure, QuTech) - To see feasible or not|
| `06_setup/` | ONLY TEST - IBM Quantum platform setup notebook - To see feasible or not|

## Notes

- Public datasets (Spambase, Lung Cancer) are **not included** — they are referenced in the thesis with download links.
- Generated results (CSVs, plots, precomputed kernels) are **not included**.
- External libraries used are listed in the thesis and in `requirements.txt` in the main project.
