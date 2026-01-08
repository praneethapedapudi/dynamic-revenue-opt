# Dynamic Revenue Optimization & Demand Elasticity Engine

A production-ready MLOps pipeline for dynamic pricing optimization using demand forecasting and price elasticity modeling. This engine enables businesses to maximize revenue by intelligently adjusting prices based on predicted demand elasticity.

## Why This Matters: Yield Management & Revenue Optimization

**Yield Management** (also known as revenue management) is a pricing strategy that allows businesses to maximize revenue by dynamically adjusting prices based on demand patterns, market conditions, and customer price sensitivity. This is widely used in industries such as:

- **Transportation**: Airlines, ride-sharing (Uber, Lyft), taxis
- **Hospitality**: Hotels, vacation rentals
- **E-commerce**: Retail pricing, flash sales
- **Entertainment**: Event tickets, streaming services

### The Business Problem

Traditional fixed pricing leaves money on the table. By understanding **price elasticity of demand** (how demand changes with price), businesses can:

1. **Increase revenue** during high-demand periods by raising prices
2. **Maintain volume** during low-demand periods by lowering prices
3. **Optimize inventory utilization** by balancing price and demand

### The Solution

This engine combines:
- **XGBoost-based demand forecasting** to predict trip demand
- **Price elasticity estimation** using log-log regression
- **Mathematical optimization** to find revenue-maximizing prices

## Architecture & Methodology

### How It Works

1. **Data Generation**: Creates realistic synthetic NYC taxi trip data with correlated features (price, weather, time, zone)

2. **Feature Engineering**: Extracts temporal features (hour, day of week, rush hour indicators) and applies cyclical encoding

3. **Demand Forecasting**: Trains an XGBoost regression model to predict demand based on:
   - Zone ID
   - Price per mile
   - Weather conditions
   - Temporal features (hour, day of week, rush hour)

4. **Elasticity Calculation**: Estimates price elasticity coefficient (β) using log-log regression:
   ```
   log(Demand) = α + β × log(Price) + ε
   ```
   Where β < 0 indicates that demand decreases as price increases (typical behavior).

5. **Revenue Optimization**: Finds the optimal price that maximizes:
   ```
   Revenue = Price × Demand
   ```
   Subject to:
   ```
   Demand = Base_Demand × (Price/Base_Price)^Elasticity
   ```
   With constraints: `0.8 × Base_Price ≤ Optimal_Price ≤ 2.0 × Base_Price`

## Project Structure

```
revenue_optimization_engine/
├── data/                      # Data artifacts (gitignored)
│   └── raw_trips.csv
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic data generation
│   ├── features.py            # Feature engineering & preprocessing
│   ├── models.py              # XGBoost model & elasticity calculation
│   └── optimization.py        # Revenue optimization logic
├── main.py                    # Pipeline orchestrator
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd revenue_optimization_engine
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

Execute the main pipeline:

```bash
python main.py
```

This will:
1. Generate 10,000 synthetic taxi trip records
2. Process and engineer features
3. Train the XGBoost demand forecasting model
4. Calculate price elasticity of demand
5. Run an optimization example for a specific scenario
6. Display revenue uplift metrics

### Expected Output

```
======================================================================
Dynamic Revenue Optimization & Demand Elasticity Engine
======================================================================

[Step 1] Generating synthetic NYC taxi data...
✓ Generated 10000 rows of data
✓ Saved to data/raw_trips.csv

[Step 2] Processing features...
✓ Training set: 8000 samples
✓ Test set: 2000 samples

[Step 3] Training demand forecasting model...
Model Performance:
  MAE: X.XX
  RMSE: X.XX
  R²: 0.XXXX

[Step 4] Calculating price elasticity...
✓ Estimated Price Elasticity: -X.XXXX

[Step 5] Running revenue optimization example...
Example Scenario:
  Zone ID: 5
  Hour: 18 (Evening Rush Hour)
  Weather: Clear
  Base Price: $X.XX per mile
  Predicted Base Demand: XXX.X trips

======================================================================
OPTIMIZATION RESULTS
======================================================================

Current Pricing:
  Price: $X.XX/mile
  Demand: XXX.X trips
  Revenue: $XXX.XX

Optimal Pricing:
  Price: $X.XX/mile
  Demand: XXX.X trips
  Revenue: $XXX.XX

Revenue Uplift: X.XX%
   ($XX.XX increase)
```

## Technical Details

### Model Architecture

- **Algorithm**: XGBoost Regressor
- **Features**: 11 engineered features (zone, price, weather, temporal)
- **Target**: Demand (integer count of trips)
- **Evaluation**: MAE, RMSE, R²

### Elasticity Estimation

Uses **log-log regression** to estimate price elasticity:
- More robust than linear regression for elasticity
- Assumes constant elasticity across price range
- Coefficient interpretation: β = -1.5 means 1% price increase → 1.5% demand decrease

### Optimization Method

Uses `scipy.optimize.minimize_scalar` with bounded constraints to find the price that maximizes revenue within feasible bounds.

## Key Metrics

- **Price Elasticity**: Measures price sensitivity (typically -1.5 to -3.0 for transportation)
- **Revenue Uplift**: Percentage increase in revenue from optimal pricing
- **Model Performance**: R², MAE, RMSE for demand forecasting accuracy

## Extending the Engine

### Future Enhancements

- **Real-time API**: Deploy as a REST API for live pricing decisions
- **Multi-product optimization**: Optimize prices across multiple zones/products simultaneously
- **Time-varying elasticity**: Model elasticity that changes with demand levels
- **Competitive pricing**: Incorporate competitor prices
- **Customer segmentation**: Different elasticity for different customer segments
- **A/B testing framework**: Validate pricing strategies

### Production Considerations

- **Model retraining**: Schedule periodic retraining with new data
- **Monitoring**: Track model drift and prediction accuracy
- **Feature store**: Centralize feature engineering for consistency
- **MLflow/DVC**: Version control for models and experiments


## License

This project is provided as-is for demonstration and educational purposes.

---

**Built with**: Python, XGBoost, scikit-learn, scipy, pandas, numpy

