# ============================================================
# Student Depression Prediction - Machine Learning Pipeline
# Based on: "Predicting Student Depression with Machine Learning"
# Models: LR, SVM, RF, AdaBoost, ANN
# Dataset: Student Depression Dataset (Kaggle, 27,902 records)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# 1. LOAD DATASET
# ============================================================
# Download from: https://www.kaggle.com/datasets/hopesb/student-depression-dataset
# Place the CSV file in the same directory and rename it to 'student_depression.csv'

print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv('student_depression.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['Depression'].value_counts()}")


# ============================================================
# 2. PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

# Drop ID column (no predictive value)
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Encode all categorical columns using LabelEncoder
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Categorical columns to encode: {categorical_cols}")
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("Encoding complete.")

# Separate features and target
X = df.drop(columns=['Depression'])
y = df['Depression']

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Train-Test Split (80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")


# ============================================================
# 4. FEATURE SCALING (required for SVM, LR, ANN)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ============================================================
# 5. CLASS BALANCING (Oversampling minority class)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Class Balancing via Oversampling")
print("=" * 60)

train_df = pd.concat([pd.DataFrame(X_train_scaled, columns=X.columns),
                       pd.Series(y_train.values, name='Depression')], axis=1)

majority = train_df[train_df['Depression'] == train_df['Depression'].value_counts().idxmax()]
minority = train_df[train_df['Depression'] == train_df['Depression'].value_counts().idxmin()]

minority_upsampled = resample(minority, replace=True,
                               n_samples=len(majority), random_state=42)
balanced_df = pd.concat([majority, minority_upsampled])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

X_train_bal = balanced_df.drop(columns=['Depression']).values
y_train_bal = balanced_df['Depression'].values

print(f"Before balancing: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"After  balancing: {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")


# ============================================================
# 6. MODEL DEFINITIONS (parameters from Table IV in paper)
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(
        random_state=42, max_iter=1000
    ),
    "SVM": SVC(
        kernel='linear', random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=50, random_state=42
    ),
    "ANN": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    ),
}


# ============================================================
# 7. TRAINING & EVALUATION FUNCTION
# ============================================================
def evaluate_models(models, X_tr, y_tr, X_te, y_te, label=""):
    results = {}
    print(f"\n{'─'*60}")
    print(f"Results {label}")
    print(f"{'─'*60}")
    print(f"{'Classifier':<22} {'Accuracy':>9} {'Recall':>9} {'Precision':>10} {'F1-Score':>10}")
    print(f"{'─'*60}")
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc  = accuracy_score(y_te, y_pred) * 100
        rec  = recall_score(y_te, y_pred, average='binary') * 100
        prec = precision_score(y_te, y_pred, average='binary') * 100
        f1   = f1_score(y_te, y_pred, average='binary') * 100
        results[name] = {'Accuracy': acc, 'Recall': rec,
                         'Precision': prec, 'F1-Score': f1, 'y_pred': y_pred}
        print(f"{name:<22} {acc:>8.2f}% {rec:>8.2f}% {prec:>9.2f}% {f1:>9.2f}%")
    return results


# ============================================================
# 8. RUN — WITHOUT BALANCING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5a: Training WITHOUT Class Balancing")
print("=" * 60)
results_unbal = evaluate_models(
    models, X_train_scaled, y_train, X_test_scaled, y_test,
    label="(Without Class Balancing)"
)

# ============================================================
# 9. RUN — WITH BALANCING
# ============================================================
print("\n" + "=" * 60)
print("STEP 5b: Training WITH Class Balancing")
print("=" * 60)
results_bal = evaluate_models(
    models, X_train_bal, y_train_bal, X_test_scaled, y_test,
    label="(With Class Balancing)"
)


# ============================================================
# 10. VISUALISATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Generating Plots")
print("=" * 60)

metrics   = ['Accuracy', 'Recall', 'Precision', 'F1-Score']
clf_names = list(models.keys())
colors    = ['#4C72B0', '#DD8452']   # blue = unbalanced, orange = balanced

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Classifier Performance: Before vs After Class Balancing', fontsize=15, fontweight='bold')

for ax, metric in zip(axes.flatten(), metrics):
    vals_unbal = [results_unbal[c][metric] for c in clf_names]
    vals_bal   = [results_bal[c][metric]   for c in clf_names]
    x = np.arange(len(clf_names))
    w = 0.35
    ax.bar(x - w/2, vals_unbal, w, label='Without Balancing', color=colors[0])
    ax.bar(x + w/2, vals_bal,   w, label='With Balancing',    color=colors[1])
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel(f'{metric} (%)')
    ax.set_ylim(60, 100)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)
    for xi, (v1, v2) in enumerate(zip(vals_unbal, vals_bal)):
        ax.text(xi - w/2, v1 + 0.3, f'{v1:.1f}', ha='center', va='bottom', fontsize=7)
        ax.text(xi + w/2, v2 + 0.3, f'{v2:.1f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: performance_comparison.png")

# --- Confusion Matrices (balanced) ---
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.suptitle('Confusion Matrices (With Class Balancing)', fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes2.flatten(), results_bal.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Not Depressed', 'Depressed'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontweight='bold')

axes2.flatten()[-1].set_visible(False)   # hide unused 6th subplot
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: confusion_matrices.png")


# ============================================================
# 11. DETAILED CLASSIFICATION REPORTS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Detailed Classification Reports (With Balancing)")
print("=" * 60)

for name, res in results_bal.items():
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    # Re-predict with the trained model
    model = models[name]
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred,
                                  target_names=['Not Depressed', 'Depressed']))

print("\n" + "=" * 60)
print("ALL DONE — plots saved to current directory.")
print("=" * 60)