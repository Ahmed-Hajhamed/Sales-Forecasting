"""
Unit tests for Sales Forecasting modules
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sales_forecasting import SalesForecaster
from data_downloader import download_sample_walmart_data

class TestSalesForecasting(unittest.TestCase):
    """Test cases for SalesForecaster class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.forecaster = SalesForecaster()
        self.sample_data = download_sample_walmart_data()
    
    def test_sample_data_generation(self):
        """Test sample data generation"""
        self.assertIsInstance(self.sample_data, pd.DataFrame)
        self.assertGreater(len(self.sample_data), 0)
        self.assertIn('Weekly_Sales', self.sample_data.columns)
        self.assertIn('Date', self.sample_data.columns)
        self.assertIn('Store', self.sample_data.columns)
        self.assertIn('Dept', self.sample_data.columns)
    
    def test_time_features_creation(self):
        """Test time-based feature creation"""
        df_with_features = self.forecaster.create_time_features(self.sample_data.head(100))
        
        # Check if time features are created
        time_features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Month_sin', 'Month_cos']
        for feature in time_features:
            self.assertIn(feature, df_with_features.columns)
        
        # Check if lag features are created
        lag_features = [col for col in df_with_features.columns if 'lag_' in col]
        self.assertGreater(len(lag_features), 0)
    
    def test_feature_preparation(self):
        """Test feature preparation for modeling"""
        df_with_features = self.forecaster.create_time_features(self.sample_data.head(100))
        X, y, df_clean = self.forecaster.prepare_features(df_with_features)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X.columns), 10)  # Should have multiple features
    
    def test_model_training(self):
        """Test model training functionality"""
        # Use smaller dataset for faster testing
        small_data = self.sample_data.head(200)
        df_with_features = self.forecaster.create_time_features(small_data)
        X, y, df_clean = self.forecaster.prepare_features(df_with_features)
        
        if len(X) > 50:  # Only test if we have enough data
            X_test, y_test, results = self.forecaster.train_models(X, y)
            
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
            
            # Check if at least one model was trained successfully
            for model_name, result in results.items():
                self.assertIn('r2', result)
                self.assertIn('mae', result)
                self.assertIn('rmse', result)

class TestDataDownloader(unittest.TestCase):
    """Test cases for data downloader functions"""
    
    def test_sample_data_structure(self):
        """Test the structure of generated sample data"""
        data = download_sample_walmart_data()
        
        # Check required columns
        required_columns = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(data['Weekly_Sales']))
        self.assertTrue(pd.api.types.is_bool_dtype(data['IsHoliday']))
        
        # Check for no negative sales
        self.assertTrue((data['Weekly_Sales'] >= 0).all())

if __name__ == '__main__':
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestSalesForecasting))
    test_suite.addTests(test_loader.loadTestsFromTestCase(TestDataDownloader))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
