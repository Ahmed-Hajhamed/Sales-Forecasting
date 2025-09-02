"""
Main Sales Forecasting Runner
This script demonstrates the complete sales forecasting pipeline
"""

import os
import sys

def main():
    """Run the complete sales forecasting analysis"""
    print("=" * 60)
    print("WALMART SALES FORECASTING PROJECT")
    print("=" * 60)
    print()
    
    print("This project implements:")
    print("✓ Time-based feature engineering (day, month, lag values)")
    print("✓ Multiple regression models for forecasting")
    print("✓ Actual vs predicted visualization over time")
    print("✓ BONUS: Rolling averages and seasonal decomposition")
    print("✓ BONUS: XGBoost/LightGBM with time-aware validation")
    print()
    
    choice = input("Choose analysis type:\n1. Basic Analysis\n2. Advanced Analysis (with all bonus features)\nEnter choice (1 or 2): ")
    
    if choice == "1":
        print("\nRunning basic sales forecasting analysis...")
        try:
            from src.sales_forecasting import SalesForecaster
            forecaster = SalesForecaster()
            forecaster.run_complete_analysis()
        except Exception as e:
            print(f"Error running basic analysis: {e}")
            
    elif choice == "2":
        print("\nRunning advanced sales forecasting analysis...")
        try:
            from src.advanced_forecasting import AdvancedSalesForecaster
            forecaster = AdvancedSalesForecaster()
            forecaster.run_complete_advanced_analysis()
        except Exception as e:
            print(f"Error running advanced analysis: {e}")
    else:
        print("Invalid choice. Running basic analysis by default...")
        try:
            from src.sales_forecasting import SalesForecaster
            forecaster = SalesForecaster()
            forecaster.run_complete_analysis()
        except Exception as e:
            print(f"Error running basic analysis: {e}")

if __name__ == "__main__":
    main()
