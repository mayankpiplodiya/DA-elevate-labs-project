# Healthcare Appointment No-Show Prediction
# Dataset: KaggleV2-May-2016.csv

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Create output folder
os.makedirs("output", exist_ok=True)


# 1. Load Dataset

file_path = "data/KaggleV2-May-2016.csv"
df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print(df.head())


# 2. Data Cleaning

# Convert date columns
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Create new features
df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['Weekday'] = df['AppointmentDay'].dt.day_name()

# Remove negative waiting days
df = df[df['WaitingDays'] >= 0]

# Convert target variable
# No-show: Yes = 1 (Missed), No = 0 (Attended)
df['No-show'] = df['No-show'].map({'Yes': 1, 'No': 0})

# Rename columns for easy handling
df.rename(columns={'No-show': 'No_show',
                   'Hipertension': 'Hypertension',
                   'Handcap': 'Handicap',
                   'Alcoholism': 'Alcoholism',
                   'SMS_received': 'SMS_received'}, inplace=True)

# Drop unnecessary columns
df.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)


# 3. Feature Engineering
# Age cleaning
df = df[df['Age'] >= 0]

# Encode categorical variables
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_weekday = LabelEncoder()
df['Weekday'] = le_weekday.fit_transform(df['Weekday'])

le_neighbourhood = LabelEncoder()
df['Neighbourhood'] = le_neighbourhood.fit_transform(df['Neighbourhood'])

# Save cleaned dataset for Tableau
df.to_csv("output/cleaned_data.csv", index=False)
print("Cleaned data saved to output/cleaned_data.csv")


# 4. Model Training
X = df.drop('No_show', axis=1)
y = df['No_show']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save predictions
pred_df = X_test.copy()
pred_df['Actual'] = y_test
pred_df['Predicted'] = y_pred
pred_df.to_csv("output/model_predictions.csv", index=False)

print("Predictions saved to output/model_predictions.csv")

# -----------------------------
# 6. Feature Importance
# -----------------------------

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

importance.to_csv("output/feature_importance.csv", index=False)

print("Feature importance saved to output/feature_importance.csv")

# -----------------------------
# 7. Simple No-show Statistics (for Tableau insight reference)
# -----------------------------

noshow_rate = df['No_show'].mean()
print("\nOverall No-show Rate:", round(noshow_rate * 100, 2), "%")
