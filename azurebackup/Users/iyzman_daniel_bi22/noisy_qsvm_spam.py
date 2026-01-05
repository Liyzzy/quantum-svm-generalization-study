import qiskit, qiskit_aer, qiskit_machine_learning
print("Qiskit:", qiskit.__version__)
print("Aer:", qiskit_aer.__version__)
print("QML:", qiskit_machine_learning.__version__)

# To ensure reproducibility of results
from qiskit_machine_learning.utils import algorithm_globals
algorithm_globals.random_seed = 12345

# --- Import Libraries ---
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, recall_score, balanced_accuracy_score

# --- Qiskit Imports ---
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC

# Load Dataset


# --- Import Spambase Column Names ---
spambase_columns = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_;",
    "char_freq_(",
    "char_freq_[",
    "char_freq_!",
    "char_freq_$",
    "char_freq_#",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
    "label"
]

# --- 1. Load the Spambase Dataset (LOCAL PATH) ---
# file_path = '/kaggle/input/spambase/spambase.data'
# file_path = r'C:\Users\User\Documents\MyProjects\FYP_ResearchProject\data\spambase\spambase.data'
file_path = "/home/azureuser/cloudfiles/code/data/spambase/spambase.data"
df = pd.read_csv(file_path, header=None, names=spambase_columns)
df.drop_duplicates(inplace=True)

print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Noise Model Factory Function


def get_noise_model(level='standard'):
    """
    Returns a noise model, backend, and pass manager for the given noise level.
    
    Noise Levels:
    - 'low': 0.01% 1q, 0.1% 2q, 0.2% readout
    - 'standard': 0.1% 1q, 1.0% 2q, 2.0% readout (realistic NISQ device)
    - 'high': 0.5% 1q, 5.0% 2q, 10.0% readout
    """
    noise_params = {
        'low': {'p_1q': 0.0001, 'p_2q': 0.001, 'p_ro': 0.002},
        'standard': {'p_1q': 0.001, 'p_2q': 0.01, 'p_ro': 0.02},
        'high': {'p_1q': 0.005, 'p_2q': 0.05, 'p_ro': 0.10},
    }
    
    params = noise_params.get(level, noise_params['standard'])
    
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(params['p_1q'], 1), ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(params['p_2q'], 2), ['cx'])
    readout_error = ReadoutError([[1 - params['p_ro'], params['p_ro']], [params['p_ro'], 1 - params['p_ro']]])
    noise_model.add_all_qubit_readout_error(readout_error)
    
    backend = AerSimulator(noise_model=noise_model, seed_simulator=12345, max_parallel_threads=16, max_parallel_experiments=0)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    
    return noise_model, backend, pm, params
    
print("Noise model factory function ready!")
print("Available levels: 'low', 'standard', 'high'")

# Experiment Configurations


# ==========================================
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


# Main Experiment


# ===================================================================
# MAIN EXPERIMENT LOOP (RESUMABLE VERSION)
# ===================================================================
import os

all_results = []

for exp_num, config in enumerate(experiments, 1):
    print("\n" + "="*80)
    print(f"EXPERIMENT {exp_num}/{len(experiments)}: {config['id']}")
    print("="*80)

    # --- CHECK IF ALREADY DONE ---
    train_file = f'kernel_train_{config["id"]}.npy'
    test_file = f'kernel_test_{config["id"]}.npy'

    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"✅ Found existing kernel files for {config['id']}. SKIPPING calculation to save time.")
        continue 
    # -----------------------------

    print(f" Samples: {config['samples']}")
    print(f" K Features: {config['k_features']}")
    print(f" Shots: {config['shots']}")
    print(f" Reps: {config['reps']}")
    print(f" Entanglement: {config['entanglement']}")
    print(f" Noise Level: {config.get('noise_level', 'standard')}")
    print("="*80)

    # ===================================================================
    # 0. GET NOISE MODEL (NEW FOR NOISY VERSION)
    # ===================================================================
    
    noise_level = config.get('noise_level', 'standard')
    noise_model, noisy_backend, pm, noise_params = get_noise_model(noise_level)
    print(f"Noise Model: 1q={noise_params['p_1q']*100:.2f}%, 2q={noise_params['p_2q']*100:.2f}%, readout={noise_params['p_ro']*100:.2f}%")

    # ===================================================================
    # 1. DATA PREPARATION
    # ===================================================================

    X = df.drop('label', axis=1)
    y = df['label']

    # --- SUBSET SELECTION (Follow ideal structure) ---
    X_subset, _, y_subset, _ = train_test_split(
        X, y,
        train_size=config['samples'],
        stratify=y,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset,
        test_size=0.30,
        random_state=42,
        stratify=y_subset
    )

    print(f"\nDataset created: {X_train.shape[0]} train, {X_test.shape[0]} test")

    # ===================================================================
    # 2. SCALING
    # ===================================================================

    # 2. VARIANCE FILTERING & SCALING (Prevents "Invalid Value" errors)
    selector_variance = VarianceThreshold(threshold=0)
    X_train_filtered = selector_variance.fit_transform(X_train)
    X_test_filtered = selector_variance.transform(X_test)
    remaining_cols = X_train.columns[selector_variance.get_support()]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered)
    X_test_scaled = scaler.transform(X_test_filtered)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=remaining_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=remaining_cols)

    print("Data scaled successfully")

    # ===================================================================
    # 3. CORRELATION-BASED FEATURE DROPPING
    # ===================================================================

    THRESH = 0.9
    corr_matrix_train = X_train_scaled_df.corr().abs()
    upper_triangle = corr_matrix_train.where(
        np.triu(np.ones(corr_matrix_train.shape), k=1).astype(bool)
    )

    columns_to_drop = set()
    for column in upper_triangle.columns:
        high_corr_partners = upper_triangle.index[upper_triangle[column] > THRESH].tolist()
        if high_corr_partners:
            for partner in high_corr_partners:
                corr_main_vs_target = y_train.corr(X_train_scaled_df[column])
                corr_partner_vs_target = y_train.corr(X_train_scaled_df[partner])

                if abs(corr_main_vs_target) < abs(corr_partner_vs_target):
                    columns_to_drop.add(column)
                else:
                    columns_to_drop.add(partner)

    to_drop_final = sorted(list(columns_to_drop))
    X_train_selected = X_train_scaled_df.drop(columns=to_drop_final)
    X_test_selected = X_test_scaled_df.drop(columns=to_drop_final)

    print(f"Dropped {len(to_drop_final)} highly correlated features")

    # ===================================================================
    # 4. SELECTKBEST
    # ===================================================================

    k_features = config['k_features']
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_kbest = selector.fit_transform(X_train_selected, y_train)
    X_test_kbest = selector.transform(X_test_selected)
    

    selected_features = X_train_selected.columns[selector.get_support()].tolist()
    print(f"SelectKBest: Selected {k_features} features:")
    for i, feat in enumerate(selected_features, 1):
        print(f" {i}. {feat}")

    # ===================================================================
    # 5. QUANTUM KERNEL SETUP (NOISY VERSION)
    # ===================================================================

    # Use NOISY sampler instead of ideal sampler
    sampler = AerSampler.from_backend(
        backend=noisy_backend,
        default_shots=config['shots']
    )

    fm = ZZFeatureMap(
        feature_dimension=k_features,
        reps=config['reps'],
        entanglement=config['entanglement']
    )

    
    fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
    qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=fm)

    print(f"Quantum kernel configured (ZZFeatureMap, reps={config['reps']}, entanglement={config['entanglement']})")

    # ===================================================================
    # 6. COMPUTE KERNEL MATRICES
    # ===================================================================

    print("\nComputing kernel matrices...")
    start_kernel = time.time()

    matrix_train = qkernel.evaluate(x_vec=X_train_kbest)
    matrix_test = qkernel.evaluate(x_vec=X_test_kbest, y_vec=X_train_kbest)

    kernel_time = time.time() - start_kernel
    print(f"Kernel computation: {kernel_time:.2f}s")

    # ===================================================================
    # 7. GRID SEARCH
    # ===================================================================

    print("\nGrid searching for optimal C...")
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        SVC(kernel='precomputed', class_weight='balanced'),
        param_grid,
        cv=cv,
        scoring='accuracy',
        verbose=0,
        n_jobs=-1
    )

    start_train = time.time()
    grid_search.fit(matrix_train, y_train)
    train_time = time.time() - start_train

    best_model = grid_search.best_estimator_
    best_c = grid_search.best_params_['C']
    cv_score = grid_search.best_score_

    print(f" → Best C: {best_c}")
    print(f" → CV Score: {cv_score:.4f}")
    print(f" → Training time: {train_time:.2f}s")

    # ===================================================================
    # 8. EVALUATION
    # ===================================================================

    y_train_pred = best_model.predict(matrix_train)
    y_test_pred = best_model.predict(matrix_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_balanced_acc = balanced_accuracy_score(y_test, y_test_pred)
    spam_recall = recall_score(y_test, y_test_pred, pos_label=1)
    gen_gap = abs(train_acc - test_acc)

    print(f" → Train Accuracy: {train_acc:.4f}")
    print(f" → Test Accuracy: {test_acc:.4f}")
    print(f" → Test Balanced Accuracy: {test_balanced_acc:.4f}")
    print(f" → Spam Recall: {spam_recall:.4f}")
    print(f" → Generalization Gap: {gen_gap:.4f}")

    # ===================================================================
    # 9. STORE RESULTS
    # ===================================================================

    all_results.append({
        'experiment_id': config['id'],
        'exp_number': exp_num,
        'samples': config['samples'],
        'k_features': k_features,
        'shots': config['shots'],
        'reps': config['reps'],
        'entanglement': config['entanglement'],
        'noise_level': noise_level,
        'selected_features': selected_features,
        'best_c': best_c,
        'cv_score': cv_score,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'test_balanced_acc': test_balanced_acc,
        'spam_recall': spam_recall,
        'gen_gap': gen_gap,
        'kernel_time': kernel_time,
        'train_time': train_time
    })

    # Print classification report
    print("\n" + "Classification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # Save kernel matrices
    np.save(f'kernel_train_{config["id"]}.npy', matrix_train)
    np.save(f'kernel_test_{config["id"]}.npy', matrix_test)
    print(f"Saved kernel matrices: kernel_train_{config['id']}.npy, kernel_test_{config['id']}.npy")

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE!")
print("="*80)


# Results Analysis & Visualization


# Convert results to DataFrame
results_df = pd.DataFrame(all_results)
print("\nExperiment Results Summary:")
print(results_df[['experiment_id', 'samples', 'k_features', 'shots', 'reps', 'noise_level', 'test_acc', 'test_balanced_acc', 'spam_recall']].to_string(index=False))

# Save results
results_df.to_csv('noisy_qsvm_results.csv', index=False)
print("\nResults saved to: noisy_qsvm_results.csv")

# ==========================================
# FIND BEST CONFIGURATIONS
# ==========================================

print("\n" + "=" * 80)
print("BEST CONFIGURATIONS")
print("=" * 80)

# Best overall test accuracy
best_acc_idx = results_df['test_acc'].idxmax()
best_acc_config = results_df.iloc[best_acc_idx]

print("\n BEST TEST ACCURACY:")
print(f"  Experiment: {best_acc_config['experiment_id']}")
print(f"  Test Accuracy: {best_acc_config['test_acc']:.4f}")
print(f"  Spam Recall: {best_acc_config['spam_recall']:.4f}")
print(f"  Gen Gap: {best_acc_config['gen_gap']:.4f}")
print(f"  Config: samples={best_acc_config['samples']}, k={best_acc_config['k_features']}, shots={best_acc_config['shots']}")

# Best spam recall (most important for spam detection)
best_recall_idx = results_df['spam_recall'].idxmax()
best_recall_config = results_df.iloc[best_recall_idx]

print("\n BEST SPAM RECALL:")
print(f"  Experiment: {best_recall_config['experiment_id']}")
print(f"  Test Accuracy: {best_recall_config['test_acc']:.4f}")
print(f"  Spam Recall: {best_recall_config['spam_recall']:.4f}")
print(f"  Gen Gap: {best_recall_config['gen_gap']:.4f}")
print(f"  Config: samples={best_recall_config['samples']}, k={best_recall_config['k_features']}, shots={best_recall_config['shots']}")

# Best generalization (lowest gap)
best_gen_idx = results_df['gen_gap'].idxmin()
best_gen_config = results_df.iloc[best_gen_idx]

print("\n BEST GENERALIZATION (Lowest Gap):")
print(f"  Experiment: {best_gen_config['experiment_id']}")
print(f"  Test Accuracy: {best_gen_config['test_acc']:.4f}")
print(f"  Spam Recall: {best_gen_config['spam_recall']:.4f}")
print(f"  Gen Gap: {best_gen_config['gen_gap']:.4f}")
print(f"  Config: samples={best_gen_config['samples']}, k={best_gen_config['k_features']}, shots={best_gen_config['shots']}")

# Best CV score (most reliable during training)
best_cv_idx = results_df['cv_score'].idxmax()
best_cv_config = results_df.iloc[best_cv_idx]

print("\n BEST CROSS-VALIDATION SCORE:")
print(f"  Experiment: {best_cv_config['experiment_id']}")
print(f"  CV Score: {best_cv_config['cv_score']:.4f}")
print(f"  Test Accuracy: {best_cv_config['test_acc']:.4f}")
print(f"  Spam Recall: {best_cv_config['spam_recall']:.4f}")
print(f"  Config: samples={best_cv_config['samples']}, k={best_cv_config['k_features']}, shots={best_cv_config['shots']}")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (18, 5)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

def plot_spambase_consistent_v2(results_data):
    """
    Generates standardized visualizations for Spambase QSVM experiments.
    Accepts either a list of result dicts or a DataFrame.
    """
    
    # Convert to DataFrame if it's a list
    if isinstance(results_data, list):
        df = pd.DataFrame(results_data)
    else:
        df = results_data.copy()

    # Map Experiment IDs to the parameter being varied
    # Based on your notebook: 
    # Exp1 = Sample Size
    # Exp2 = Dimensionality (Features)
    # Exp3 = Shot Noise
    # Exp4 = Reps (Circuit Depth)
    # Exp5 = Entanglement
    
    exp_map = {
        'Exp1': {'param': 'samples',      'xlabel': 'Training Samples',       'log_x': False, 'type': 'line'},
        'Exp2': {'param': 'k_features',   'xlabel': 'Feature Dimension (Qubits)', 'log_x': False, 'type': 'line'},
        'Exp3': {'param': 'shots',        'xlabel': 'Shots (Measurement)',    'log_x': True,  'type': 'line'},
        'Exp4': {'param': 'reps',         'xlabel': 'Circuit Depth (Reps)',   'log_x': False, 'type': 'bar'},
        'Exp5': {'param': 'entanglement', 'xlabel': 'Entanglement Type',      'log_x': False, 'type': 'bar'}
    }

    # Iterate through each defined experiment group
    for group_key, config in exp_map.items():
        # Filter data for this experiment group (matching ID string)
        subset = df[df['experiment_id'].astype(str).str.contains(group_key)].copy()
        
        if subset.empty:
            print(f"Skipping {group_key}: No data found.")
            continue
            
        param = config['param']
        
        # Ensure correct data types for plotting
        if param == 'samples' or param == 'k_features' or param == 'shots' or param == 'reps':
            subset[param] = pd.to_numeric(subset[param])
            
        # Sort by parameter to ensure line plots are ordered correctly
        if config['type'] == 'line':
            subset = subset.sort_values(by=param)
            
        print(f"\n--- Plotting {group_key}: Effect of {config['xlabel']} ---")
        
        fig, axes = plt.subplots(1, 3)
        
        # -------------------------------------------------------
        # SUBPLOT 1: PERFORMANCE (Accuracy vs Recall)
        # -------------------------------------------------------
        if config['type'] == 'bar':
            x_pos = np.arange(len(subset))
            width = 0.35
            # Plot bars
            axes[0].bar(x_pos - width/2, subset['test_acc'], width, label='Test Accuracy', color='#1f77b4', alpha=0.9)
            axes[0].bar(x_pos + width/2, subset['spam_recall'], width, label='Spam Recall', color='#ff7f0e', alpha=0.9)
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(subset[param])
        else:
            # Plot lines
            axes[0].plot(subset[param], subset['test_acc'], 'o-', label='Test Accuracy', color='#1f77b4')
            axes[0].plot(subset[param], subset['spam_recall'], 's--', label='Spam Recall', color='#ff7f0e')
            if config['log_x']: axes[0].set_xscale('log', base=2)

        axes[0].set_xlabel(config['xlabel'])
        axes[0].set_ylabel('Score')
        axes[0].set_title(f'Performance vs {config["param"].title()}')
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)

        # -------------------------------------------------------
        # SUBPLOT 2: GENERALIZATION GAP (Overfitting)
        # -------------------------------------------------------
        if config['type'] == 'bar':
            axes[1].bar(range(len(subset)), subset['gen_gap'], color='#d62728', alpha=0.7)
            axes[1].set_xticks(range(len(subset)))
            axes[1].set_xticklabels(subset[param])
        else:
            axes[1].plot(subset[param], subset['gen_gap'], 'D-', color='#d62728')
            if config['log_x']: axes[1].set_xscale('log', base=2)
            
        axes[1].set_xlabel(config['xlabel'])
        axes[1].set_ylabel('Gap (Train Acc - Test Acc)')
        axes[1].set_title('Generalization Gap (Lower is Better)')
        axes[1].grid(True, alpha=0.3)

        # -------------------------------------------------------
        # SUBPLOT 3: COMPUTATIONAL COST (Time)
        # -------------------------------------------------------
        if config['type'] == 'bar':
            axes[2].bar(range(len(subset)), subset['kernel_time'], color='#2ca02c', alpha=0.7)
            axes[2].set_xticks(range(len(subset)))
            axes[2].set_xticklabels(subset[param])
        else:
            axes[2].plot(subset[param], subset['kernel_time'], '^-', color='#2ca02c')
            if config['log_x']: axes[2].set_xscale('log', base=2)

        axes[2].set_xlabel(config['xlabel'])
        axes[2].set_ylabel('Kernel Calculation Time (s)')
        axes[2].set_title('Computational Cost')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------
    # SUMMARY HEATMAP
    # -------------------------------------------------------
    print("\n--- Global Performance Overview ---")
    plt.figure(figsize=(14, 10))
    
    # Metrics to display in heatmap
    # Check if columns exist before selecting
    avail_metrics = [m for m in ['test_acc', 'spam_recall', 'test_balanced_acc', 'gen_gap', 'cv_score'] if m in df.columns]
    
    # Prepare data
    heatmap_data = df.set_index('experiment_id')[avail_metrics]
    
    # Plot
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.75, linewidths=.5, cbar_kws={'label': 'Score'})
    
    plt.title('Experiment Summary Heatmap', fontsize=16, pad=20)
    plt.ylabel('Experiment ID')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
# Note: In your notebook, the results are stored in the list 'all_results'
# If 'all_results' is populated, this will work directly.

try:
    if 'all_results' in locals() and len(all_results) > 0:
        print("Using in-memory 'all_results' list...")
        plot_spambase_consistent_v2(all_results)
    elif 'results_df' in locals():
        print("Using in-memory 'results_df' dataframe...")
        plot_spambase_consistent_v2(results_df)
    else:
        # Fallback: Try to read CSV if available
        print("Attempting to read from CSV...")
        df_temp = pd.read_csv('qsvm_selectkbest_all_experiments.csv')
        plot_spambase_consistent_v2(df_temp)
        
except Exception as e:
    print(f"Could not generate plots: {e}")
    print("Ensure 'all_results' list is populated or 'qsvm_selectkbest_all_experiments.csv' exists.")
