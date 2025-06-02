#!/usr/bin/env python3
"""
Cancer Classification Pipeline
Double CV experiment with 6 different models
"""

# ========== 1. library分段 ==========
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, 
                           recall_score, precision_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ========== 2. 全局变量分段 ==========
# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Data paths
DATA_ROOT = os.getenv('DATA_ROOT', '.')
DATA_PATH = os.path.join(DATA_ROOT, 'Cancer2025exam.csv')

# Cross-validation parameters
N_F = 10  # Number of outer CV folds

# Model names
MODEL_SET = ['FS_PCA_NN', 'FS_PCA_SVM', 'RF', 'FS_PCA_QDA', 'FS_PCA_KNN', 'FS_PCA_LR']

# Output directories
OUTPUT_ROOT = os.path.join(DATA_ROOT, 'output')
METRICS_DIR = os.path.join(OUTPUT_ROOT, 'metrics')
INDICES_DIR = os.path.join(OUTPUT_ROOT, 'indices')
PARAMS_DIR = os.path.join(OUTPUT_ROOT, 'parameters')
PLOTS_DIR = os.path.join(OUTPUT_ROOT, 'plots')
PREDICTIONS_DIR = os.path.join(OUTPUT_ROOT, 'predictions')
PROBABILITIES_DIR = os.path.join(OUTPUT_ROOT, 'probabilities')

# Create output directories
for dir_path in [OUTPUT_ROOT, METRICS_DIR, INDICES_DIR, PARAMS_DIR, 
                 PLOTS_DIR, PREDICTIONS_DIR, PROBABILITIES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Hyperparameter grids for ParameterSampler
PARAM_GRIDS = {
    'FS_PCA_NN': {
        'K': [10, 30, 50, 70, 100, 200],  # Number of features after filter
        'N_dim': [5, 10, 20, 50, 100, None],    # PCA dimensions
        'hidden_layer_sizes': [(50, 50), (100, 50)],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500, 1000, 2000]  # Added training iterations
    },
    'FS_PCA_SVM': {
        'K': [10, 30, 50, 70, 100, 200],
        'N_dim': [5, 10, 20, 50, 100, None],
        'C': [0.1, 1, 10, 100]
    },
    'RF': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'FS_PCA_QDA': {
        'K': [10, 30, 50, 70, 100, 200],
        'N_dim': [5, 10, 20, 50, 100, None],
        'reg_param': [0.0, 0.01, 0.1, 0.5]
    },
    'FS_PCA_KNN': {
        'K': [10, 30, 50, 70, 100, 200],
        'N_dim': [5, 10, 20, 50, 100, None],
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance']
    },
    'FS_PCA_LR': {
        'K': [10, 30, 50, 70, 100, 200],
        'N_dim': [5, 10, 20, 50, 100, None],
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'max_iter': [500, 1000, 2000]
    }
}

# Number of parameter samples per model
N_PARAM_SAMPLES = 20

# ========== 3. 函数分段 ==========

def load_data(path):
    """Load data from CSV file"""
    return pd.read_csv(path)

def get_class_distribution(y):
    """Get distribution of classes"""
    return dict(Counter(y))

def normalize_features(X_train, X_test):
    """Normalize features using StandardScaler fitted on training data"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def apply_feature_selection(X_train, y_train, X_test, k):
    """Apply feature selection using f_classif"""
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected

def apply_pca(X_train, X_test, n_components):
    """Apply PCA transformation"""
    if n_components is None:
        return X_train, X_test  # Skip PCA
    pca = PCA(n_components=min(n_components, X_train.shape[1]))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def get_model(model_name, params):
    """Get model instance based on name and parameters"""
    if model_name == 'FS_PCA_NN':
        return MLPClassifier(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
            alpha=params.get('alpha', 0.0001),
            learning_rate_init=params.get('learning_rate_init', 0.001),
            max_iter=params.get('max_iter', 1000),  # Make max_iter tunable
            random_state=RANDOM_SEED
        )
    elif model_name == 'FS_PCA_SVM':
        return SVC(
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            probability=True,
            random_state=RANDOM_SEED
        )
    elif model_name == 'RF':
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=RANDOM_SEED
        )
    elif model_name == 'FS_PCA_QDA':
        return QuadraticDiscriminantAnalysis(
            reg_param=params.get('reg_param', 0.0)
        )
    elif model_name == 'FS_PCA_KNN':
        return KNeighborsClassifier(
            n_neighbors=params.get('n_neighbors', 5),
            weights=params.get('weights', 'uniform')
        )
    elif model_name == 'FS_PCA_LR':
        return LogisticRegression(
            C=params.get('C', 1.0),
            penalty=params.get('penalty', 'l2'),
            max_iter=params.get('max_iter', 1000),
            random_state=RANDOM_SEED
        )

def preprocess_and_fit(X_train, y_train, X_test, model_name, params):
    """Preprocess data and fit model based on pipeline requirements"""
    # Normalize features
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    
    if model_name != 'RF':
        # Apply feature selection
        k = params.get('K', 50)
        X_train_fs, X_test_fs = apply_feature_selection(X_train_norm, y_train, X_test_norm, k)
        
        # Apply PCA
        n_dim = params.get('N_dim', None)
        X_train_final, X_test_final = apply_pca(X_train_fs, X_test_fs, n_dim)
    else:
        # RF uses all normalized features
        X_train_final, X_test_final = X_train_norm, X_test_norm
    
    # Get and fit model
    model = get_model(model_name, params)
    model.fit(X_train_final, y_train)
    
    return model, X_train_final, X_test_final

def calculate_metrics(y_true, y_pred, labels):
    """Calculate all required metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'per_class_f1': f1_score(y_true, y_pred, average=None, labels=labels),
        'per_class_recall': recall_score(y_true, y_pred, average=None, labels=labels),
        'per_class_precision': precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels)
    }
    
    # Calculate per-class specificity
    cm = metrics['confusion_matrix']
    specificities = []
    for i in range(len(labels)):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    metrics['per_class_specificity'] = np.array(specificities)
    
    return metrics

def save_metrics(metrics, model_name, fold_idx):
    """Save metrics to files"""
    # Save main metrics
    main_metrics = {
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'macro_f1': metrics['macro_f1']
    }
    
    for metric_name, value in main_metrics.items():
        filename = f"{metric_name}_{model_name}_{fold_idx}.npy"
        np.save(os.path.join(METRICS_DIR, filename), value)
    
    # Save per-class metrics
    per_class_metrics = {
        'per_class_f1': metrics['per_class_f1'],
        'per_class_recall': metrics['per_class_recall'],
        'per_class_specificity': metrics['per_class_specificity'],
        'per_class_precision': metrics['per_class_precision']
    }
    
    for metric_name, values in per_class_metrics.items():
        filename = f"{metric_name}_{model_name}_{fold_idx}.npy"
        np.save(os.path.join(METRICS_DIR, filename), values)

# ========== 4. 数据加载分段 ==========
print("Loading data...")
df = load_data(DATA_PATH)
print(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# ========== 5. 数据集相关变量定义分段 ==========
# Separate features and labels
X = df.iloc[:, 1:].values  # Features are from column 2 onwards
y = df.iloc[:, 0].values   # Labels are in the first column

# Dataset characteristics
dim_feat = X.shape[1]  # Original dimension of features
num_samples = X.shape[0]  # Total number of samples
unique_labels = np.unique(y)
num_labels = len(unique_labels)  # Number of label classes
label_distribution = get_class_distribution(y)

print(f"Feature dimension: {dim_feat}")
print(f"Number of samples: {num_samples}")
print(f"Number of labels: {num_labels}")
print(f"Label distribution: {label_distribution}")

# ========== 5.5 为每个样本添加唯一ID分段 ==========
# Create sample IDs based on row index in original data
sample_ids = np.arange(num_samples)
print(f"Sample IDs created: {sample_ids[0]} to {sample_ids[-1]}")

# ========== 6. normalization分段 ==========
# Check if normalization is needed by examining feature scales
feature_ranges = np.ptp(X, axis=0)  # peak-to-peak (max - min) for each feature
print(f"Feature range statistics: min={np.min(feature_ranges):.2f}, max={np.max(feature_ranges):.2f}")
if np.max(feature_ranges) / np.min(feature_ranges) > 10:
    print("Normalization is needed due to different feature scales")
else:
    print("Features have similar scales, but normalization will still be applied for consistency")

# ========== 7. 实验数据采样方式分段 ==========
# Find the class with minimum samples
min_class_size = min(label_distribution.values())
print(f"Minimum class size: {min_class_size}")

# Sampling strategy selection
SAMPLING_STRATEGY = 2  # 1: Use all data, 2: Undersample to min_class_size

if SAMPLING_STRATEGY == 1:
    print("Strategy 1: Using all data without undersampling")
    X_exp = X
    y_exp = y
    sample_ids_exp = sample_ids
elif SAMPLING_STRATEGY == 3: ################这里故意写错防止忘记将sampling模式改回1.这里要改回2
    print(f"Strategy 2: Undersampling each class to {min_class_size} samples")
    sampled_indices = []
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        sampled_indices.extend(np.random.choice(label_indices, min_class_size, replace=False))
    
    sampled_indices = np.array(sampled_indices)
    X_exp = X[sampled_indices]
    y_exp = y[sampled_indices]
    sample_ids_exp = sample_ids[sampled_indices]

print(f"Experimental data shape: {X_exp.shape}")

# ========== 8. 待评估模型管线集合分段 ==========
print(f"\nModels to evaluate: {MODEL_SET}")

# ========== 9. 上述模型对应的待优化超参数网格分段 ==========
print("\nParameter grids defined for each model")

# ========== 10. double CV 分段 ==========
print("\n" + "="*50)
print("Starting Double Cross-Validation")
print("="*50)

# Outer CV
outer_cv = StratifiedKFold(n_splits=N_F, shuffle=True, random_state=RANDOM_SEED)

# Store results for all models and folds
all_test_scores = {model: [] for model in MODEL_SET}
all_test_predictions = {model: [] for model in MODEL_SET}
all_test_probabilities = {model: [] for model in MODEL_SET}
all_best_params = {model: [] for model in MODEL_SET}

# Outer fold loop
for outer_fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(X_exp, y_exp)):
    print(f"\n--- Outer Fold {outer_fold_idx + 1}/{N_F} ---")
    
    # Split data
    X_train_val, X_test = X_exp[train_val_idx], X_exp[test_idx]
    y_train_val, y_test = y_exp[train_val_idx], y_exp[test_idx]
    test_sample_ids = sample_ids_exp[test_idx]
    
    # ========== 11. 存储测试样本索引 ==========
    test_indices_filename = f"test_index_{outer_fold_idx + 1}.npy"
    np.save(os.path.join(INDICES_DIR, test_indices_filename), test_sample_ids)
    print(f"Saved test indices to {test_indices_filename}")
    
    # Inner CV for each model
    for model_name in MODEL_SET:
        print(f"\nEvaluating {model_name}...")
        
        # Sample hyperparameters
        param_sampler = ParameterSampler(
            PARAM_GRIDS[model_name], 
            n_iter=N_PARAM_SAMPLES, 
            random_state=RANDOM_SEED
        )
        param_list = list(param_sampler)
        
        # Store scores for each parameter combination
        param_scores = []
        
        # Inner CV
        inner_cv = StratifiedKFold(n_splits=N_F, shuffle=True, random_state=RANDOM_SEED)
        
        for param_idx, params in enumerate(param_list):
            inner_scores = []
            
            for inner_fold_idx, (train_idx, val_idx) in enumerate(inner_cv.split(X_train_val, y_train_val)):
                # Split inner data
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
                
                try:
                    # Train model with current parameters
                    model, _, X_val_processed = preprocess_and_fit(X_train, y_train, X_val, model_name, params)
                    
                    # Predict and evaluate
                    y_pred = model.predict(X_val_processed)
                    score = balanced_accuracy_score(y_val, y_pred)
                    inner_scores.append(score)
                except Exception as e:
                    print(f"Error with params {params}: {e}")
                    inner_scores.append(0.0)
            
            # Average score across inner folds
            mean_score = np.mean(inner_scores)
            param_scores.append((mean_score, params))
        
        # Find best parameters
        best_score, best_params = max(param_scores, key=lambda x: x[0])
        print(f"Best params for {model_name}: {best_params} (score: {best_score:.4f})")
        
        # ========== 11. 存储最佳参数 ==========
        best_params_filename = f"best_para_{model_name}_{outer_fold_idx + 1}.npy"
        np.save(os.path.join(PARAMS_DIR, best_params_filename), best_params)
        all_best_params[model_name].append(best_params)
        
        # Train final model on full train-val set with best params
        final_model, _, X_test_processed = preprocess_and_fit(
            X_train_val, y_train_val, X_test, model_name, best_params
        )
        
        # Make predictions
        y_pred = final_model.predict(X_test_processed)
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, unique_labels)
        all_test_scores[model_name].append(metrics)
        
        # ========== 11. 存储评估指标 ==========
        save_metrics(metrics, model_name, outer_fold_idx + 1)
        
        # ========== 11. 存储预测结果 ==========
        pred_results = np.column_stack((test_sample_ids, y_pred))
        pred_filename = f"pred_res_{model_name}_{outer_fold_idx + 1}.npy"
        np.save(os.path.join(PREDICTIONS_DIR, pred_filename), pred_results)
        
        # ========== 11. 存储预测概率（除SVM外） ==========
        if model_name != 'FS_PCA_SVM' and hasattr(final_model, 'predict_proba'):
            y_proba = final_model.predict_proba(X_test_processed)
            prob_results = np.column_stack((test_sample_ids.reshape(-1, 1), y_proba))
            prob_filename = f"pred_dist_{model_name}_{outer_fold_idx + 1}.npy"
            np.save(os.path.join(PROBABILITIES_DIR, prob_filename), prob_results)

print("\n" + "="*50)
print("Double CV completed!")
print("="*50)

# ========== 12. 可视化分段 ==========
print("\nGenerating visualizations...")

# 1. Box plot of metrics across all folds
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = ['accuracy', 'balanced_accuracy', 'macro_f1']
metric_names = ['Accuracy', 'Balanced Accuracy', 'Macro-F1']

for idx, (metric_key, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
    data_to_plot = []
    labels_to_plot = []
    
    for model_name in MODEL_SET:
        scores = [score[metric_key] for score in all_test_scores[model_name]]
        data_to_plot.append(scores)
        labels_to_plot.append(model_name.replace('_', ' '))
    
    axes[idx].boxplot(data_to_plot, labels=labels_to_plot)
    axes[idx].set_title(f'Boxplot of {metric_name} distributions across {N_F} CV tests')
    axes[idx].set_ylabel(metric_name)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'metrics_boxplot.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Row-normalized mean confusion matrix
# Average confusion matrices across folds
mean_conf_matrices = {}
for model_name in MODEL_SET:
    conf_matrices = [score['confusion_matrix'] for score in all_test_scores[model_name]]
    mean_conf_matrices[model_name] = np.mean(conf_matrices, axis=0)

# Create subplot for each model's confusion matrix
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, model_name in enumerate(MODEL_SET):
    cm = mean_conf_matrices[model_name]
    # Row normalize
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels,
                ax=axes[idx])
    axes[idx].set_title(f'{model_name.replace("_", " ")}')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

plt.suptitle(f'Row-normalized mean confusion matrix over {N_F} tests', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Per-class metrics table
print("\nGenerating per-class metrics tables...")
for model_name in MODEL_SET:
    # Collect per-class metrics across folds
    f1_scores = np.array([score['per_class_f1'] for score in all_test_scores[model_name]])
    recall_scores = np.array([score['per_class_recall'] for score in all_test_scores[model_name]])
    specificity_scores = np.array([score['per_class_specificity'] for score in all_test_scores[model_name]])
    precision_scores = np.array([score['per_class_precision'] for score in all_test_scores[model_name]])
    
    # Calculate mean and std
    metrics_table = pd.DataFrame({
        'Class': unique_labels,
        'F1_mean': np.mean(f1_scores, axis=0),
        'F1_std': np.std(f1_scores, axis=0),
        'Recall_mean': np.mean(recall_scores, axis=0),
        'Recall_std': np.std(recall_scores, axis=0),
        'Specificity_mean': np.mean(specificity_scores, axis=0),
        'Specificity_std': np.std(specificity_scores, axis=0),
        'Precision_mean': np.mean(precision_scores, axis=0),
        'Precision_std': np.std(precision_scores, axis=0)
    })
    
    # Save table
    table_filename = f'per_class_metrics_{model_name}.csv'
    metrics_table.to_csv(os.path.join(PLOTS_DIR, table_filename), index=False)
    print(f"Saved per-class metrics for {model_name}")

# 4. Box plot of prediction confidence (max predicted probability)
print("\nGenerating prediction confidence plots...")
models_with_proba = [m for m in MODEL_SET if m != 'FS_PCA_SVM']

for model_name in models_with_proba:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    confidence_data = []
    fold_labels = []
    
    for fold_idx in range(N_F):
        # Load probability predictions
        prob_filename = f"pred_dist_{model_name}_{fold_idx + 1}.npy"
        prob_path = os.path.join(PROBABILITIES_DIR, prob_filename)
        
        if os.path.exists(prob_path):
            prob_data = np.load(prob_path)
            probabilities = prob_data[:, 1:]  # Exclude sample ID column
            max_probs = np.max(probabilities, axis=1)
            confidence_data.append(max_probs)
            fold_labels.append(f'Fold {fold_idx + 1}')
    
    if confidence_data:
        ax.boxplot(confidence_data, labels=fold_labels)
        ax.set_title(f'Distribution of Prediction Confidence for {model_name.replace("_", " ")} across {N_F} tests')
        ax.set_xlabel('Test Folder Index')
        ax.set_ylabel('Maximum Predicted Probability')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'confidence_{model_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

# 5. Calibration curves
print("\nGenerating calibration curves...")
from sklearn.calibration import calibration_curve

for model_name in models_with_proba:
    fig, axes = plt.subplots(1, N_F, figsize=(20, 4))
    if N_F == 1:
        axes = [axes]
    
    for fold_idx in range(N_F):
        # Load probability predictions and true labels
        prob_filename = f"pred_dist_{model_name}_{fold_idx + 1}.npy"
        prob_path = os.path.join(PROBABILITIES_DIR, prob_filename)
        
        # Get test indices to match with true labels
        test_indices = np.load(os.path.join(INDICES_DIR, f"test_index_{fold_idx + 1}.npy"))
        
        if os.path.exists(prob_path):
            prob_data = np.load(prob_path)
            probabilities = prob_data[:, 1:]  # Exclude sample ID column
            
            # Get true labels for this fold
            test_mask = np.isin(sample_ids_exp, test_indices)
            y_true_fold = y_exp[test_mask]
            
            # For multi-class, we'll create calibration curves for each class
            for class_idx, class_label in enumerate(unique_labels):
                # Binary classification: class vs rest
                y_binary = (y_true_fold == class_label).astype(int)
                prob_positive = probabilities[:, class_idx]
                
                # Calculate calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_binary, prob_positive, n_bins=5
                )
                
                axes[fold_idx].plot(mean_predicted_value, fraction_of_positives, 
                                   marker='o', label=f'Class {class_label}')
            
            # Perfect calibration line
            axes[fold_idx].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            axes[fold_idx].set_xlabel('Mean Predicted Probability')
            axes[fold_idx].set_ylabel('Fraction of Positives')     
            axes[fold_idx].set_title(f'Fold {fold_idx + 1}')
            axes[fold_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[fold_idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Calibration Curves for {model_name.replace("_", " ")}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'calibration_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print("\n" + "="*50)
print("Pipeline execution completed!")
print(f"All results saved to: {OUTPUT_ROOT}")
print("="*50)