import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


DATA_PATH = os.path.join("data", "digital_wallet_ltv_dataset.csv")
OUTPUT_PATH = os.path.join("output", "predicted_ltv_output.csv")


def load_data(path):
    df = pd.read_csv(path)
    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df


def preprocess_data(df):

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df


def train_model(df):

    X = df.drop(columns=['LTV'])
    y = df['LTV']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 3))

    return model


def predict_and_segment(model, df):

    df['Predicted_LTV'] = model.predict(df.drop(columns=['LTV']))

    df['LTV_Segment'] = pd.qcut(
        df['Predicted_LTV'],
        q=3,
        labels=['Low Value', 'Medium Value', 'High Value']
    )

    return df


def save_output(df, path):

    if not os.path.exists("output"):
        os.makedirs("output")

    df.to_csv(path, index=False)
    print("\nOutput saved at:", path)


def plot_feature_importance(model, df):

    importance = model.feature_importances_
    features = df.drop(columns=['LTV', 'Predicted_LTV', 'LTV_Segment']).columns

    plt.figure()
    plt.bar(features, importance)
    plt.xticks(rotation=90)
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


def main():

    df = load_data(DATA_PATH)

    df = preprocess_data(df)

    model = train_model(df)

    df = predict_and_segment(model, df)

    save_output(df, OUTPUT_PATH)

    plot_feature_importance(model, df)


if __name__ == "__main__":
    main()