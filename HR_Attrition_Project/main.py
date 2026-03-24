
# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import shap

# 2. LOAD DATASET

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 3. DATA PREPROCESSING


# Convert target variable
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop unnecessary columns
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Save original dataset for Tableau
df_original = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df_original.to_csv("outputs/hr_clean.csv", index=False)

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)


# 4. EXPLORATORY DATA ANALYSIS

# Attrition distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Attrition Distribution")
plt.show()

# Salary vs Attrition
plt.figure(figsize=(6,4))
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title("Salary vs Attrition")
plt.show()

# Years at company vs Attrition
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='YearsAtCompany', hue='Attrition', kde=True)
plt.title("Years at Company vs Attrition")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 5. MODEL BUILDING


X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. LOGISTIC REGRESSION

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


# 7. DECISION TREE
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d')
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# 8. FEATURE IMPORTANCE (Decision Tree)

feature_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()

# 9. SHAP ANALYSIS

# SHAP for Logistic Regression
explainer = shap.Explainer(log_model, X_train)
shap_values = explainer(X_test)

print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_values, X_test)


# 10. SAVE MODEL (OPTIONAL)

import joblib

joblib.dump(log_model, "outputs/logistic_model.pkl")
joblib.dump(dt_model, "outputs/dt_model.pkl")

print("\nModels saved successfully!")

# 11. FINAL MESSAGE

print("\nProject Execution Completed Successfully!")