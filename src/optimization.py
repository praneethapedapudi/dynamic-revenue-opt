"""
Revenue Optimization Module
Finds optimal pricing to maximize revenue given demand elasticity.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def revenue_function(price, base_demand, elasticity, base_price):
    """
    Calculate revenue for a given price.
    
    Revenue = Price × Demand
    
    Where Demand follows: Demand = Base_Demand × (Price/Base_Price)^Elasticity
    
    Parameters:
    -----------
    price : float
        New price to evaluate
    base_demand : float
        Baseline demand at base_price
    elasticity : float
        Price elasticity coefficient (typically negative)
    base_price : float
        Baseline price
    
    Returns:
    --------
    float
        Revenue at the given price
    """
    # Calculate demand at new price using elasticity formula
    price_ratio = price / base_price
    demand = base_demand * (price_ratio ** elasticity)
    
    # Revenue = Price × Demand
    revenue = price * demand
    
    return revenue


def optimize_price(base_demand, elasticity, base_price, min_multiplier=0.8, max_multiplier=2.0):
    """
    Find the optimal price that maximizes revenue.
    
    Uses scipy.optimize to find the price within constraints that maximizes
    Revenue = Price × (Base_Demand × (Price/Base_Price)^Elasticity)
    
    Parameters:
    -----------
    base_demand : float
        Baseline demand at base_price
    elasticity : float
        Price elasticity coefficient (typically negative)
    base_price : float
        Baseline/current price
    min_multiplier : float
        Minimum price multiplier (default: 0.8 = 80% of base)
    max_multiplier : float
        Maximum price multiplier (default: 2.0 = 200% of base)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - optimal_price: The price that maximizes revenue
        - optimal_demand: Demand at optimal price
        - optimal_revenue: Maximum revenue
        - base_revenue: Revenue at base price
        - revenue_uplift: Percentage increase in revenue
    """
    # Define price bounds
    price_min = base_price * min_multiplier
    price_max = base_price * max_multiplier
    
    # Negative revenue function for minimization (we want to maximize)
    def neg_revenue(price):
        return -revenue_function(price, base_demand, elasticity, base_price)
    
    # Optimize
    result = minimize_scalar(
        neg_revenue,
        bounds=(price_min, price_max),
        method='bounded'
    )
    
    optimal_price = result.x
    optimal_demand = base_demand * ((optimal_price / base_price) ** elasticity)
    optimal_revenue = revenue_function(optimal_price, base_demand, elasticity, base_price)
    
    # Calculate baseline metrics
    base_revenue = revenue_function(base_price, base_demand, elasticity, base_price)
    revenue_uplift = ((optimal_revenue - base_revenue) / base_revenue) * 100
    
    return {
        'optimal_price': optimal_price,
        'optimal_demand': optimal_demand,
        'optimal_revenue': optimal_revenue,
        'base_price': base_price,
        'base_demand': base_demand,
        'base_revenue': base_revenue,
        'revenue_uplift_pct': revenue_uplift
    }


if __name__ == "__main__":
    # Example usage
    print("Testing revenue optimization...")
    
    # Example parameters
    base_demand = 100
    elasticity = -1.5  # Elastic demand
    base_price = 3.0
    
    result = optimize_price(base_demand, elasticity, base_price)
    
    print(f"\nOptimization Results:")
    print(f"Base Price: ${result['base_price']:.2f}")
    print(f"Optimal Price: ${result['optimal_price']:.2f}")
    print(f"Base Demand: {result['base_demand']:.1f}")
    print(f"Optimal Demand: {result['optimal_demand']:.1f}")
    print(f"Base Revenue: ${result['base_revenue']:.2f}")
    print(f"Optimal Revenue: ${result['optimal_revenue']:.2f}")
    print(f"Revenue Uplift: {result['revenue_uplift_pct']:.2f}%")

