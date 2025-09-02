"""
Sales Forecasting Package
========================

A comprehensive sales forecasting solution implementing time series analysis,
feature engineering, and multiple regression models for predicting future sales.

Modules:
--------
- sales_forecasting: Basic forecasting implementation
- advanced_forecasting: Advanced implementation with bonus features
- data_downloader: Data loading and sample generation utilities

Author: Sales Forecasting Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Sales Forecasting Team"

from .sales_forecasting import SalesForecaster
from .advanced_forecasting import AdvancedSalesForecaster
from .data_downloader import load_walmart_data

__all__ = [
    'SalesForecaster',
    'AdvancedSalesForecaster', 
    'load_walmart_data'
]
