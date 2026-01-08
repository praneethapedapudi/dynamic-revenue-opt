"""
Dynamic Revenue Optimization & Demand Elasticity Engine
Main entry point for the revenue optimization pipeline.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_generator import generate_mock_data
from src.features import load_and_process_data
from src.models import ElasticityPredictor
from src.optimization import optimize_price
import pandas as pd
import numpy as np


def main():
    """
    Main pipeline orchestrator:
    1. Generate synthetic data
    2. Process and engineer features
    3. Train demand forecasting model
    4. Calculate price elasticity
    5. Run optimization example
    6. Display results
    """
    print("=" * 70)
    print("Dynamic Revenue Optimization & Demand Elasticity Engine")
    print("=" * 70)
    
    # Step 1: Generate Data
    print("\n[Step 1] Generating synthetic NYC taxi data...")
    df = generate_mock_data(n_rows=10000)
    data_path = os.path.join('data', 'raw_trips.csv')
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"✓ Generated {len(df)} rows of data")
    print(f"✓ Saved to {data_path}")
    
    # Step 2: Process Features
    print("\n[Step 2] Processing features...")
    X_train, X_test, y_train, y_test = load_and_process_data(data_path)
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train Model
    print("\n[Step 3] Training demand forecasting model...")
    predictor = ElasticityPredictor(random_state=42)
    predictor.train(X_train, y_train)
    
    # Evaluate model
    metrics = predictor.evaluate(X_test, y_test)
    print(f"\nModel Performance:")
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    # Step 4: Calculate Elasticity
    print("\n[Step 4] Calculating price elasticity...")
    elasticity = predictor.calculate_elasticity(X_train, y_train)
    print(f"\n✓ Estimated Price Elasticity: {elasticity:.4f}")
    
    # Step 5: Optimization Example
    print("\n[Step 5] Running revenue optimization example...")
    
    # Select a specific scenario (e.g., Zone 5, Hour 18 - evening rush)
    example_zone = 5
    example_hour = 18
    example_weather = 0  # Clear
    
    # Find a sample from training data matching this scenario
    mask = (
        (X_train['zone_id'] == example_zone) &
        (X_train['hour'] == example_hour) &
        (X_train['weather_condition'] == example_weather)
    )
    
    if mask.sum() > 0:
        sample_idx = X_train[mask].index[0]
        base_price = X_train.loc[sample_idx, 'price_per_mile']
        base_demand = predictor.predict(X_train.loc[[sample_idx]])[0]
    else:
        # Fallback: use median values
        base_price = X_train['price_per_mile'].median()
        # Create a sample row for prediction
        sample_row = X_train.iloc[0].copy()
        sample_row['zone_id'] = example_zone
        sample_row['hour'] = example_hour
        sample_row['weather_condition'] = example_weather
        base_demand = predictor.predict(pd.DataFrame([sample_row]))[0]
    
    print(f"\nExample Scenario:")
    print(f"  Zone ID: {example_zone}")
    print(f"  Hour: {example_hour} (Evening Rush Hour)")
    print(f"  Weather: {'Clear' if example_weather == 0 else 'Rain'}")
    print(f"  Base Price: ${base_price:.2f} per mile")
    print(f"  Predicted Base Demand: {base_demand:.1f} trips")
    
    # Run optimization
    optimization_result = optimize_price(
        base_demand=base_demand,
        elasticity=elasticity,
        base_price=base_price,
        min_multiplier=0.8,
        max_multiplier=2.0
    )
    
    # Step 6: Display Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"\nCurrent Pricing:")
    print(f"  Price: ${optimization_result['base_price']:.2f}/mile")
    print(f"  Demand: {optimization_result['base_demand']:.1f} trips")
    print(f"  Revenue: ${optimization_result['base_revenue']:.2f}")
    
    print(f"\nOptimal Pricing:")
    print(f"  Price: ${optimization_result['optimal_price']:.2f}/mile")
    print(f"  Demand: {optimization_result['optimal_demand']:.1f} trips")
    print(f"  Revenue: ${optimization_result['optimal_revenue']:.2f}")
    
    print(f"\n📈 Revenue Uplift: {optimization_result['revenue_uplift_pct']:.2f}%")
    print(f"   (${optimization_result['optimal_revenue'] - optimization_result['base_revenue']:.2f} increase)")
    
    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

