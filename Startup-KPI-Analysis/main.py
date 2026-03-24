# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import os

# 2. LOAD DATA

def load_data():
    df = pd.read_csv("data/50_Startups.csv")
    print("✅ Data Loaded Successfully")
    return df


# 3. PROCESS DATA (KPI CALCULATION)
def process_data(df):
    # Rename Profit → Revenue
    df.rename(columns={"Profit": "Revenue"}, inplace=True)

    # Create Customers (assumption)
    df['Customers'] = (df['Marketing Spend'] / 1000).astype(int) + 1

    # CAC
    df['CAC'] = df['Marketing Spend'] / df['Customers']

    # LTV
    df['LTV'] = df['Revenue'] / df['Customers']

    # Expenses
    df['Expenses'] = df['R&D Spend'] + df['Administration'] + df['Marketing Spend']

    # Burn Rate
    df['Burn_Rate'] = df['Expenses'] - df['Revenue']

    # Run Rate (Annual)
    df['Run_Rate'] = df['Revenue'] * 12

    # LTV:CAC Ratio
    df['LTV_CAC_Ratio'] = df['LTV'] / df['CAC']

    print("✅ KPI Calculations Done")
    return df

# 4. COHORT ANALYSIS

def cohort_analysis(df):
    df['Month'] = pd.date_range(start='2023-01-01', periods=len(df), freq='ME')
    df['Cohort'] = df['Month'].dt.to_period('M')

    print("✅ Cohort Analysis Done")
    return df

# 5. SAVE OUTPUT

def save_output(df):
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/startup_kpi_output.csv", index=False)

    print("✅ CSV Saved in /output folder")


# 6. UPLOAD TO GOOGLE SHEETS

def upload_to_sheets(df):
    import gspread
    from google.oauth2.service_account import Credentials

    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    # Load credentials
    creds = Credentials.from_service_account_file(
        "credentials/credentials.json",
        scopes=scope
    )

    client = gspread.authorize(creds)

    # Open your sheet (must exist)
    sheet = client.open("Startup KPI Dashboard").sheet1

    # Clear old data
    sheet.clear()

    # Upload new data
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

    print("✅ Data uploaded to Google Sheets!")

# 7. MAIN FUNCTION

def main():
    df = load_data()
    df = process_data(df)
    df = cohort_analysis(df)
    save_output(df)

    # Upload to Google Sheets (optional if credentials ready)
    try:
        upload_to_sheets(df)
    except Exception as e:
        print("⚠ Google Sheets Upload Skipped:", e)

    print("\n🚀 PROJECT COMPLETED SUCCESSFULLY!")

# RUN
if __name__ == "__main__":
    main()