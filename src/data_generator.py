"""
Synthetic NYC Taxi Data Generator
Creates realistic mock data for revenue optimization modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_mock_data(n_rows=10000):
    """
    Generate synthetic NYC taxi trip data with realistic correlations.
    
    Parameters:
    -----------
    n_rows : int
        Number of rows to generate (default: 10000)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: pickup_datetime, zone_id, price_per_mile, 
        weather_condition, demand
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate base datetime range (last 30 days, hourly intervals)
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    # Generate random pickup datetimes
    time_deltas = np.random.uniform(0, (end_date - start_date).total_seconds(), n_rows)
    pickup_datetime = [start_date + timedelta(seconds=td) for td in time_deltas]
    
    # Zone IDs (1 to 15)
    zone_id = np.random.randint(1, 16, n_rows)
    
    # Base price per mile (random between $2.0 and $5.0)
    price_per_mile = np.random.uniform(2.0, 5.0, n_rows)
    
    # Weather condition (0=Clear, 1=Rain) - weighted towards clear
    weather_condition = np.random.choice([0, 1], n_rows, p=[0.7, 0.3])
    
    # Extract hour from datetime for demand correlation
    hours = pd.Series(pickup_datetime).dt.hour.values
    
    # Generate demand with realistic correlations:
    # - Negative correlation with price (higher price -> lower demand)
    # - Positive correlation with rain (rain -> higher demand)
    # - Higher demand during rush hours (7-9 AM, 5-7 PM)
    is_rush_hour = ((hours >= 7) & (hours <= 9)) | ((hours >= 17) & (hours <= 19))
    rush_hour_multiplier = np.where(is_rush_hour, 1.5, 1.0)
    
    # Base demand calculation
    base_demand = 50 + (hours * 2)  # Higher demand during day hours
    price_effect = -8 * price_per_mile  # Negative price elasticity
    weather_effect = weather_condition * 15  # Rain increases demand
    rush_effect = (rush_hour_multiplier - 1) * 20
    
    demand = base_demand + price_effect + weather_effect + rush_effect
    
    # Add some noise and ensure positive demand
    demand = demand + np.random.normal(0, 10, n_rows)
    demand = np.maximum(demand, 1).astype(int)  # Ensure at least 1 trip
    
    # Create DataFrame
    df = pd.DataFrame({
        'pickup_datetime': pickup_datetime,
        'zone_id': zone_id,
        'price_per_mile': price_per_mile,
        'weather_condition': weather_condition,
        'demand': demand
    })
    
    return df


if __name__ == "__main__":
    # Generate and save data
    print("Generating synthetic taxi data...")
    df = generate_mock_data(n_rows=10000)
    output_path = "../data/raw_trips.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())

