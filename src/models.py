"""
Demand Forecasting and Price Elasticity Models
Uses XGBoost for demand prediction and calculates price elasticity.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import xgboost as xgb


class ElasticityPredictor:
    """
    Predicts demand using XGBoost and calculates price elasticity of demand.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ElasticityPredictor.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.elasticity_coefficient = None
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train XGBoost model to predict demand.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target (demand)
        """
        print("Training XGBoost demand forecasting model...")
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print("Model training completed!")
    
    def predict(self, X):
        """
        Predict demand for given features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
        
        Returns:
        --------
        np.ndarray
            Predicted demand values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test set.
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target (demand)
        
        Returns:
        --------
        dict
            Dictionary with MAE, RMSE, and R2 scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation!")
        
        y_pred = self.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        return metrics
    
    def calculate_elasticity(self, X_train, y_train):
        """
        Calculate price elasticity of demand using Log-Log regression.
        
        Price Elasticity = % Change in Demand / % Change in Price
        
        Uses log-log model: log(Demand) = α + β * log(Price) + ε
        where β is the elasticity coefficient.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features (must include 'price_per_mile')
        y_train : pd.Series
            Training target (demand)
        
        Returns:
        --------
        float
            Price elasticity coefficient (typically negative)
        """
        print("\nCalculating price elasticity of demand...")
        
        # Extract price and demand
        price = X_train['price_per_mile'].values
        demand = y_train.values
        
        # Remove zeros and negatives for log transformation
        valid_mask = (price > 0) & (demand > 0)
        price_clean = price[valid_mask]
        demand_clean = demand[valid_mask]
        
        # Log-Log transformation
        log_price = np.log(price_clean)
        log_demand = np.log(demand_clean)
        
        # Linear regression: log(demand) = α + β * log(price)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_price, log_demand
        )
        
        self.elasticity_coefficient = slope
        
        print(f"Elasticity Coefficient (β): {self.elasticity_coefficient:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        print(f"P-value: {p_value:.4e}")
        
        if self.elasticity_coefficient < 0:
            print("✓ Negative elasticity confirmed (demand decreases with price)")
        else:
            print("⚠ Warning: Positive elasticity (unusual for demand)")
        
        return self.elasticity_coefficient

