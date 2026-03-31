
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuration
spambase_columns = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
    "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
    "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
    "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
    "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
    "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
    "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
    "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
    "capital_run_length_longest", "capital_run_length_total", "label"
]

# Load Data
file_path = r'c:\Users\User\Documents\MyProjects\AI_Projects\quantum-svm-generalization-study\notebooks\eda\spambase.csv'
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# Drop duplicates
print(f"Original shape: {df.shape}")
df.drop_duplicates(inplace=True)
print(f"Shape after dropping duplicates: {df.shape}")

# Prepare X and y
X_full = df.drop('label', axis=1)
y_full = df['label']

# Split
print("Splitting data...")
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full,
    train_size=0.70,
    random_state=42,
    stratify=y_full
)
print(f"Train samples: {len(X_train_full)}, Test samples: {len(X_test_full)}")

# Preprocessing Pipeline
print("Running preprocessing pipeline...")

# 1. Variance Threshold
selector_variance = VarianceThreshold(threshold=0)
X_train_filtered = selector_variance.fit_transform(X_train_full)
X_test_filtered = selector_variance.transform(X_test_full)
remaining_cols = X_train_full.columns[selector_variance.get_support()]

# 2. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=remaining_cols)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=remaining_cols)

# 3. Correlation Filtering
print("Running correlation filtering...")
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
            corr_main = y_train_full.corr(X_train_scaled_df[column])
            corr_partner = y_train_full.corr(X_train_scaled_df[partner])
            if abs(corr_main) < abs(corr_partner):
                columns_to_drop.add(column)
            else:
                columns_to_drop.add(partner)

X_train_selected = X_train_scaled_df.drop(columns=sorted(list(columns_to_drop)))
X_test_selected = X_test_scaled_df.drop(columns=sorted(list(columns_to_drop)))
print(f"Features after correlation filtering: {X_train_selected.shape[1]}")

# 4. Feature Selection (SelectKBest)
k_features = 14
print(f"Selecting top {k_features} features using Mutual Info...")
selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
X_train_final = selector.fit_transform(X_train_selected, y_train_full)
X_test_final = selector.transform(X_test_selected)

selected_features = X_train_selected.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")

# 5. Training
print("Starting GridSearchCV for RBF Kernel...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}
svm_model = SVC(kernel='rbf', random_state=42)

grid_search = GridSearchCV(
    svm_model, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1
)

start_train = time.time()
grid_search.fit(X_train_final, y_train_full)
train_time = time.time() - start_train

best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test_final)
y_train_pred = best_model.predict(X_train_final)

# Calculate Metrics
train_acc = accuracy_score(y_train_full, y_train_pred)
test_acc = accuracy_score(y_test_full, y_test_pred)
precision = precision_score(y_test_full, y_test_pred)
recall = recall_score(y_test_full, y_test_pred)
f1 = f1_score(y_test_full, y_test_pred)
gen_gap = train_acc - test_acc

print("\n" + "="*50)
print("RESULTS (Full 14 Features)")
print("="*50)
print(f"Samples: {len(df)}")
print(f"Features: {k_features}")
print(f"Best Params: {grid_search.best_params_}")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Generalization Gap: {gen_gap:.4f}")
print(f"Kernel Time (Train Time): {train_time:.4f} seconds")
print("="*50)
