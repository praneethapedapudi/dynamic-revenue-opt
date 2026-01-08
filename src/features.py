"""
Feature Engineering Module
Cleans data and creates features for demand forecasting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_process_data(csv_path='../data/raw_trips.csv'):
    """
    Load CSV data and create engineered features.
    
    Parameters:
    -----------
    csv_path : str
        Path to the raw trips CSV file
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) DataFrames
    """
    # Load data
    df = pd.read_csv(csv_path)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Feature Engineering
    # Extract temporal features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    
    # Rush hour indicator (7-9 AM, 5-7 PM)
    df['is_rush_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) | 
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for hour (sine/cosine)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Select features for modeling
    feature_columns = [
        'zone_id',
        'price_per_mile',
        'weather_condition',
        'hour',
        'day_of_week',
        'is_rush_hour',
        'is_weekend',
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos'
    ]
    
    X = df[feature_columns].copy()
    y = df['demand'].copy()
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the feature engineering
    print("Loading and processing data...")
    X_train, X_test, y_train, y_test = load_and_process_data()
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"\nFeature columns: {list(X_train.columns)}")
    print(f"\nFirst few training rows:")
    print(X_train.head())
    print(f"\nTarget statistics:")
    print(y_train.describe())

