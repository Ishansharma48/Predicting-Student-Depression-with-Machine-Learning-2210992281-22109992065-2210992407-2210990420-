# =========================================================
# ADVANCED STUDENT DEPRESSION PREDICTION SYSTEM
# =========================================================

# Import Libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv("student_depression_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# ---------------------------------------------------------
# LABEL ENCODING
# ---------------------------------------------------------

encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = encoder.fit_transform(df[col])

# ---------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------

# Correlation Heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# Depression Distribution
plt.figure(figsize=(6,5))
sns.countplot(x='Depression', data=df)
plt.title("Depression Class Distribution")
plt.show()

# Academic Pressure vs Depression
plt.figure(figsize=(8,5))
sns.boxplot(x='Depression', y='Academic Pressure', data=df)
plt.title("Academic Pressure vs Depression")
plt.show()

# Sleep Duration Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Sleep Duration'], kde=True)
plt.title("Sleep Duration Distribution")
plt.show()

# ---------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------

X = df.drop("Depression", axis=1)
y = df["Depression"]

# ---------------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# MACHINE LEARNING MODELS
# ---------------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Support Vector Machine":
        SVC(probability=True),

    "Random Forest":
        RandomForestClassifier(n_estimators=200),

    "Decision Tree":
        DecisionTreeClassifier(),

    "AdaBoost":
        AdaBoostClassifier(n_estimators=100),

    "Gradient Boosting":
        GradientBoostingClassifier(),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Artificial Neural Network":
        MLPClassifier(
            hidden_layer_sizes=(128,64),
            max_iter=500,
            random_state=42
        )
}

# ---------------------------------------------------------
# MODEL TRAINING & EVALUATION
# ---------------------------------------------------------

accuracy_list = []

for name, model in models.items():

    print("\n================================================")
    print("MODEL:", name)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    accuracy_list.append(accuracy)

    print("Accuracy :", round(accuracy*100,2), "%")
    print("Precision:", round(precision*100,2), "%")
    print("Recall   :", round(recall*100,2), "%")
    print("F1 Score :", round(f1*100,2), "%")

    # Classification Report
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues')

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ---------------------------------------------------------
# MODEL ACCURACY COMPARISON
# ---------------------------------------------------------

model_names = list(models.keys())

comparison_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracy_list
})

print("\n================ MODEL COMPARISON ================")
print(comparison_df)

# Accuracy Graph
plt.figure(figsize=(12,6))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=comparison_df,
    palette="viridis"
)

plt.xticks(rotation=20)
plt.title("Accuracy Comparison of ML Models")
plt.ylim(0,1)
plt.show()

# ---------------------------------------------------------
# BEST MODEL SELECTION
# ---------------------------------------------------------

best_index = np.argmax(accuracy_list)
best_model_name = model_names[best_index]

print("\nBest Performing Model:", best_model_name)
print("Best Accuracy:",
      round(accuracy_list[best_index]*100,2), "%")

# ---------------------------------------------------------
# FEATURE IMPORTANCE USING RANDOM FOREST
# ---------------------------------------------------------

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)

importance = rf.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features")
print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(12,8))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_importance,
    palette="magma"
)

plt.title("Feature Importance for Depression Prediction")
plt.show()

# ---------------------------------------------------------
# ROC CURVE
# ---------------------------------------------------------

best_model = rf

y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label='ROC curve (AUC = %0.2f)' % roc_auc
)

plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Depression Prediction')

plt.legend(loc="lower right")
plt.show()

# ---------------------------------------------------------
# HYPERPARAMETER TUNING
# ---------------------------------------------------------

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("Best Accuracy:",
      round(grid_search.best_score_ * 100, 2), "%")

# ---------------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------------

import joblib

joblib.dump(best_model, "student_depression_model.pkl")

print("\nModel Saved Successfully!")