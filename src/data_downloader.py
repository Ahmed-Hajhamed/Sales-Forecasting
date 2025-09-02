"""
Walmart Sales Data Downloader and Processor
This script helps download and process the Walmart Sales Forecasting dataset
"""

import pandas as pd
import numpy as np
import os
import requests
from io import StringIO

def download_sample_walmart_data():
    """
    Download or create sample Walmart sales data
    Since direct Kaggle download requires API setup, we'll create realistic sample data
    """
    print("Creating realistic Walmart sales sample data...")
    
    # Create 3 years of daily data for multiple stores and departments
    date_range = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    
    data = []
    stores = [1, 2, 3, 4, 5]  # 5 stores
    departments = [1, 2, 3, 7, 8, 13, 16, 20, 24, 26]  # Popular departments
    
    for store in stores:
        for dept in departments:
            for date in date_range:
                # Base sales with store and department effects
                base_sales = 1000 + store * 100 + dept * 50
                
                # Seasonal effects
                month = date.month
                day_of_week = date.dayofweek
                
                # Holiday effect (simplified)
                is_holiday = (
                    (month == 12 and date.day in [24, 25, 31]) or  # Christmas/New Year
                    (month == 11 and date.day >= 22 and date.day <= 28) or  # Thanksgiving week
                    (month == 7 and date.day == 4) or  # Independence Day
                    (month == 1 and date.day == 1)  # New Year
                )
                
                # Seasonal multiplier
                seasonal_mult = 1 + 0.3 * np.sin(2 * np.pi * (month - 1) / 12)
                
                # Weekend effect
                weekend_mult = 1.2 if day_of_week in [5, 6] else 1.0
                
                # Holiday boost
                holiday_mult = 1.5 if is_holiday else 1.0
                
                # Random noise
                noise = np.random.normal(1, 0.1)
                
                # Calculate final sales
                weekly_sales = base_sales * seasonal_mult * weekend_mult * holiday_mult * noise
                weekly_sales = max(0, weekly_sales)  # Ensure non-negative
                
                # Economic indicators (simplified)
                temperature = 50 + 30 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 5)
                fuel_price = 3.0 + 0.5 * np.random.random() + 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
                cpi = 200 + (date.year - 2021) * 5 + np.random.normal(0, 2)
                unemployment = 6 + 2 * np.random.random()
                
                data.append({
                    'Store': store,
                    'Dept': dept,
                    'Date': date,
                    'Weekly_Sales': round(weekly_sales, 2),
                    'IsHoliday': is_holiday,
                    'Temperature': round(temperature, 1),
                    'Fuel_Price': round(fuel_price, 3),
                    'CPI': round(cpi, 2),
                    'Unemployment': round(unemployment, 2)
                })
    
    df = pd.DataFrame(data)
    return df

def download_real_walmart_data():
    """
    Instructions for downloading real Walmart data from Kaggle
    """
    print("To download the real Walmart sales dataset:")
    print("1. Go to: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting")
    print("2. Download the 'train.csv' file")
    print("3. Place it in this directory")
    print("4. The script will automatically use it if present")
    print()

def load_walmart_data():
    """
    Load Walmart data - either from Kaggle download or generate sample
    """
    # Check if real Kaggle data exists
    if os.path.exists('../data/train.csv'):
        print("Found Kaggle Walmart dataset (train.csv). Loading...")
        df = pd.read_csv('../data/train.csv')
        print(f"Loaded {len(df)} records from Kaggle dataset")
        return df
    elif os.path.exists('../data/walmart_sales_data.csv'):
        print("Found existing sample dataset. Loading...")
        df = pd.read_csv('../data/walmart_sales_data.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        print("No existing dataset found. Creating sample data...")
        download_real_walmart_data()
        df = download_sample_walmart_data()
        
        # Save the sample data
        os.makedirs('../data', exist_ok=True)
        df.to_csv('../data/walmart_sales_data.csv', index=False)
        print(f"Created and saved sample dataset with {len(df)} records")
        return df

if __name__ == "__main__":
    # Download/create the data
    data = load_walmart_data()
    print("\nDataset Info:")
    print(f"Shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Stores: {sorted(data['Store'].unique())}")
    print(f"Departments: {sorted(data['Dept'].unique())}")
    print("\nFirst few rows:")
    print(data.head())
    print("\nDataset ready for sales forecasting!")
