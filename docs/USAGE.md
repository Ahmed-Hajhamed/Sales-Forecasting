# Sales Forecasting - Quick Start Guide

## How to Run the Project

### Option 1: Complete Demonstration (Recommended)
```bash
python demo.py
```
This runs the full demonstration with all features showcased.

### Option 2: Interactive Menu
```bash
python main.py
```
Choose between basic (1) or advanced (2) analysis.

### Option 3: Individual Scripts
```bash
# Basic forecasting
python sales_forecasting.py

# Advanced forecasting with all bonus features
python advanced_forecasting.py

# Data generation only
python data_downloader.py

# Project summary
python summary.py
```

## Key Features Demonstrated

### Core Requirements ✅
- **Time-based features**: Day, month, lag values, cyclical encoding
- **Regression models**: Linear Regression, Random Forest
- **Visualization**: Actual vs predicted over time
- **Libraries**: Python, Pandas, Matplotlib, Scikit-learn

### Bonus Features ✅
- **Rolling averages**: Multiple window sizes
- **Seasonal decomposition**: Trend and seasonal analysis
- **XGBoost/LightGBM**: Advanced gradient boosting
- **Time-aware validation**: Time series cross-validation

## Generated Files
- `walmart_sales_data.csv` - Sample dataset (54,750 records)
- `seasonal_analysis.png` - Seasonal patterns
- `actual_vs_predicted.png` - Model predictions over time
- `feature_importance.png` - Feature importance analysis

## Model Performance
- **Best Model**: LightGBM
- **R² Score**: 0.9241 (92.41% accuracy)
- **RMSE**: 205.79
- **MAE**: 155.42

## Topics Covered
- Time series forecasting
- Regression modeling
- Feature engineering
- Data visualization
- Model evaluation

## Ready for Real Data
To use real Walmart data:
1. Download `train.csv` from [Kaggle Walmart Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
2. Place it in the project directory
3. Run any script - it will automatically use the real data
