import json
import os

# Define the new experiments list as a string to be injected
noisy_experiments_code = """# ==========================================
# EXPERIMENT CONFIGURATIONS
# ==========================================

experiments = [
    # --- EXP 1: Sample Size Effect (Generalization) ---
    {'id': 'Exp1_100samp',  'samples': 100, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp1_200samp',  'samples': 200, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp1_300samp',  'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp1_500samp',  'samples': 500, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},

    # --- EXP 2: Dimensionality Effect (Quantum Complexity) ---
    {'id': 'Exp2_2feat', 'samples': 300, 'k_features': 2, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp2_4feat', 'samples': 300, 'k_features': 4, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp2_6feat', 'samples': 300, 'k_features': 6, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp2_8feat', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp2_10feat', 'samples': 300, 'k_features': 10, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp2_12feat', 'samples': 300, 'k_features': 12, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},

    # --- EXP 3: Shot Noise Effect (Measurement Precision) ---
    {'id': 'Exp3_128shots', 'samples': 300, 'k_features': 8, 'shots': 128, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp3_512shots', 'samples': 300, 'k_features': 8, 'shots': 512, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp3_1024shots', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},

    # --- EXP 4: Reps Effect (NEW: Circuit Complexity) ---
    {'id': 'Exp4_Reps1', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp4_Reps2', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 2, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp4_Reps3', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 3, 'entanglement': 'linear', 'noise_level': 'standard'},

    # --- EXP 5: Entanglement Ablation (NEW) ---
    {'id': 'Exp5_Linear', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp5_Circular', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'circular', 'noise_level': 'standard'},
    {'id': 'Exp5_Full', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'full', 'noise_level': 'standard'},

    # --- EXP 6: Noise Ablation (NEW - Noise-specific experiments) ---
    {'id': 'Exp6_LowNoise', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'low'},
    {'id': 'Exp6_StdNoise', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'standard'},
    {'id': 'Exp6_HighNoise', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear', 'noise_level': 'high'},
]

print(f"Total experiments configured: {len(experiments)}")
print("Running might take hours or even several days")
"""

em_experiments_code = """# ==========================================
# EXPERIMENT CONFIGURATIONS (Same as Ideal/Noisy)
# ==========================================
# ZNE uses Richardson extrapolation with scale factors [1.0, 3.0]

experiments = [
    # --- EXP 1: Sample Size Effect (Generalization) ---
    {'id': 'Exp1_100samp',  'samples': 100, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp1_200samp',  'samples': 200, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp1_300samp',  'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp1_500samp',  'samples': 500, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},

    # --- EXP 2: Dimensionality Effect (Quantum Complexity) ---
    {'id': 'Exp2_2feat',   'samples': 300, 'k_features': 2,  'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp2_4feat',   'samples': 300, 'k_features': 4,  'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp2_6feat',  'samples': 300, 'k_features': 6, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp2_8feat',  'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp2_10feat',  'samples': 300, 'k_features': 10, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp2_12feat',  'samples': 300, 'k_features': 12, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},

    # --- EXP 3: Shot Noise Effect (Measurement Precision) ---
    {'id': 'Exp3_128shots',  'samples': 300, 'k_features': 8, 'shots': 128,  'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp3_512shots',  'samples': 300, 'k_features': 8, 'shots': 512,  'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp3_1024shots', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},

    # --- EXP 4: Reps Effect (Circuit Complexity) ---
    {'id': 'Exp4_Reps1', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1,  'entanglement': 'linear'},  
    {'id': 'Exp4_Reps2', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 2,  'entanglement': 'linear'},  
    {'id': 'Exp4_Reps3', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 3,  'entanglement': 'linear'},

    # --- EXP 5: Entanglement Ablation ---
    {'id': 'Exp5_Linear',  'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'linear'},
    {'id': 'Exp5_Circular', 'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'circular'},
    {'id': 'Exp5_Full',    'samples': 300, 'k_features': 8, 'shots': 1024, 'reps': 1, 'entanglement': 'full'},
]

print(f"Total experiments configured: {len(experiments)}")
print("Each experiment runs ZNE with scale factors [1.0, 3.0] - Richardson extrapolation")
print("Running might take hours or even several days (2x kernel computations per experiment)")
"""

def update_notebook(filepath, new_code, key_string):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if key_string in source:
                cell['source'] = new_code.splitlines(keepends=True)
                found = True
                print(f"Updated cell in {filepath}")
                break
    
    if not found:
        print(f"Warning: Could not find cell to update in {filepath}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=None, ensure_ascii=False) # indent=None to keep it compact-ish or consistent

# File paths
noisy_path = r"C:\Users\User\Documents\MyProjects\AI_Projects\quantum-svm-generalization-study\models\noisy-qsvm\noisy_qsvm_spambase_2.ipynb"
em_path = r"C:\Users\User\Documents\MyProjects\AI_Projects\quantum-svm-generalization-study\models\mitigated-qsvm\em-qsvm-spambase.ipynb"

update_notebook(noisy_path, noisy_experiments_code, "experiments = [")
update_notebook(em_path, em_experiments_code, "experiments = [")
print("Done")
