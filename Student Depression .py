# Predicting Student Depression using Machine Learning

# Import Libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

# Replace with your dataset path
df = pd.read_csv("student_depression_dataset.csv")

# Display first 5 rows
print(df.head())

# ---------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------

# Check missing values
print(df.isnull().sum())

# Fill missing values if any
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = label_encoder.fit_transform(df[column])

# ---------------------------------------------------
# Feature Selection
# ---------------------------------------------------

# Target Variable
y = df['Depression']

# Features
X = df.drop('Depression', axis=1)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ---------------------------------------------------
# Train Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# Initialize Models
# ---------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Artificial Neural Network": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42
    )
}

# ---------------------------------------------------
# Train and Evaluate Models
# ---------------------------------------------------

results = {}

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print("\n===================================")
    print(f"Model: {name}")
    print("Accuracy:", round(accuracy * 100, 2), "%")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ---------------------------------------------------
# Compare Model Performance
# ---------------------------------------------------

results_df = pd.DataFrame({
    'Model': results.keys(),
    'Accuracy': results.values()
})

print("\nModel Comparison:")
print(results_df)

# Plot Accuracy Comparison
plt.figure(figsize=(10,6))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')

plt.title("Machine Learning Model Accuracy Comparison")
plt.xticks(rotation=15)
plt.ylim(0,1)
plt.show()

# ---------------------------------------------------
# Best Model
# ---------------------------------------------------

best_model = max(results, key=results.get)

print("\nBest Performing Model:", best_model)
print("Highest Accuracy:", round(results[best_model] * 100, 2), "%")