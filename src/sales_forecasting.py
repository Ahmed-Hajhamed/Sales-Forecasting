"""
Sales Forecasting with Walmart Sales Data
Task-7: Complete implementation with time series features, regression models, and advanced techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available. Will use alternative models.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not available. Will use alternative models.")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Statsmodels not available. Will skip seasonal decomposition.")

class SalesForecaster:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_sample_data(self, n_samples=2000):
        """Generate sample Walmart-like sales data"""
        print("Generating sample Walmart sales data...")
        
        # Create date range
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # Generate base sales with trends and seasonality
        trend = np.linspace(1000, 1500, n_samples)
        seasonal = 200 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)  # Yearly seasonality
        weekly = 100 * np.sin(2 * np.pi * np.arange(n_samples) / 7)  # Weekly seasonality
        noise = np.random.normal(0, 50, n_samples)
        
        # Special events (holidays, promotions)
        special_events = np.zeros(n_samples)
        # Random promotional spikes
        promo_days = np.random.choice(n_samples, size=n_samples//20, replace=False)
        special_events[promo_days] = np.random.normal(300, 100, len(promo_days))
        
        # Generate stores and departments
        stores = np.random.choice(range(1, 46), n_samples)  # 45 stores like Walmart dataset
        departments = np.random.choice(range(1, 100), n_samples)  # Various departments
        
        # Calculate sales
        sales = trend + seasonal + weekly + special_events + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative sales
        
        # Create DataFrame
        data = pd.DataFrame({
            'Date': dates,
            'Store': stores,
            'Dept': departments,
            'Weekly_Sales': sales,
            'IsHoliday': np.random.choice([True, False], n_samples, p=[0.05, 0.95]),
            'Temperature': np.random.normal(70, 20, n_samples),
            'Fuel_Price': np.random.normal(3.5, 0.5, n_samples),
            'CPI': np.random.normal(200, 10, n_samples),
            'Unemployment': np.random.normal(7, 2, n_samples)
        })
        
        return data
    
    def create_time_features(self, df):
        """Create time-based features"""
        print("Creating time-based features...")
        
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclical features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        # Sort by date for lag features
        df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 28]:
            df[f'Sales_lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
        
        # Rolling averages
        for window in [7, 14, 28]:
            df[f'Sales_rolling_mean_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'Sales_rolling_std_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
        
        # Sales growth features
        df['Sales_growth_1'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.pct_change(1))
        df['Sales_growth_7'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.pct_change(7))
        
        return df
    
    def seasonal_decomposition(self, df, store_id=1, dept_id=1):
        """Perform seasonal decomposition for a specific store-department combination"""
        if not HAS_STATSMODELS:
            print("Statsmodels not available. Skipping seasonal decomposition.")
            return
            
        print(f"Performing seasonal decomposition for Store {store_id}, Dept {dept_id}...")
        
        # Filter data for specific store and department
        subset = df[(df['Store'] == store_id) & (df['Dept'] == dept_id)].copy()
        subset = subset.sort_values('Date').set_index('Date')
        
        if len(subset) < 104:  # Need at least 2 years of weekly data
            print("Not enough data for seasonal decomposition")
            return
            
        # Perform decomposition
        decomposition = seasonal_decompose(subset['Weekly_Sales'], model='additive', period=52)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        decomposition.observed.plot(ax=axes[0], title='Original Sales')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        plt.tight_layout()
        plt.savefig('outputs/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return decomposition
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("Preparing features for modeling...")
        
        # Select feature columns
        feature_cols = [
            'Store', 'Dept', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear',
            'WeekOfYear', 'Quarter', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 
            'DayOfWeek_cos', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'
        ]
        
        # Add lag and rolling features if they exist
        lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col or 'growth_' in col]
        feature_cols.extend(lag_cols)
        
        # Filter feature columns to only include those that exist in the dataframe
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Convert boolean to int
        if 'IsHoliday' in df.columns:
            df = df.copy()
            df['IsHoliday'] = df['IsHoliday'].astype(int)
        
        # Remove rows with NaN values (due to lag features)
        df_clean = df[feature_cols + ['Weekly_Sales']].dropna()
        
        if len(df_clean) == 0:
            print("Warning: No data left after removing NaN values. Using simpler feature set.")
            # Use basic features only
            basic_features = ['Store', 'Dept', 'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear',
                             'WeekOfYear', 'Quarter', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 
                             'DayOfWeek_cos', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
            basic_features = [col for col in basic_features if col in df.columns]
            df_clean = df[basic_features + ['Weekly_Sales']].dropna()
            feature_cols = basic_features
        
        X = df_clean[feature_cols]
        y = df_clean['Weekly_Sales']
        
        self.feature_names = feature_cols
        print(f"Using {len(feature_cols)} features for modeling")
        print(f"Training on {len(X)} samples")
        
        return X, y, df_clean
    
    def train_models(self, X, y):
        """Train multiple models"""
        print("Training models...")
        
        # Split data with time-aware approach
        # Use last 20% for testing
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        if HAS_XGB:
            models_to_train['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, 
                random_state=42,
                verbosity=0
            )
            
        if HAS_LGB:
            models_to_train['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100, 
                random_state=42,
                verbosity=-1
            )
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            if name in ['Linear Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        
        self.models = results
        return X_test, y_test, results
    
    def plot_results(self, X_test, y_test, results, df_with_features):
        """Plot actual vs predicted values"""
        print("Creating visualizations...")
        
        # Get test dates for plotting - use original df with features
        split_point = int(len(df_with_features) * 0.8)
        test_dates = df_with_features.iloc[split_point:]['Date'].values[:len(y_test)]
        
        # Create subplots
        n_models = len(results)
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, result) in enumerate(results.items()):
            y_pred = result['predictions']
            
            # Time series plot
            axes[idx, 0].plot(test_dates, y_test.values, label='Actual', alpha=0.7)
            axes[idx, 0].plot(test_dates, y_pred, label='Predicted', alpha=0.7)
            axes[idx, 0].set_title(f'{name} - Time Series')
            axes[idx, 0].set_xlabel('Date')
            axes[idx, 0].set_ylabel('Sales')
            axes[idx, 0].legend()
            axes[idx, 0].tick_params(axis='x', rotation=45)
            
            # Scatter plot
            axes[idx, 1].scatter(y_test, y_pred, alpha=0.5)
            axes[idx, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[idx, 1].set_xlabel('Actual Sales')
            axes[idx, 1].set_ylabel('Predicted Sales')
            axes[idx, 1].set_title(f'{name} - Actual vs Predicted (R² = {result["r2"]:.4f})')
        
        plt.tight_layout()
        plt.savefig('outputs/sales_forecasting_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance for tree-based models
        self.plot_feature_importance(results)
    
    def plot_feature_importance(self, results):
        """Plot feature importance for tree-based models"""
        tree_models = ['Random Forest', 'XGBoost', 'LightGBM']
        available_tree_models = [name for name in tree_models if name in results]
        
        if not available_tree_models:
            return
            
        fig, axes = plt.subplots(1, len(available_tree_models), figsize=(6*len(available_tree_models), 8))
        if len(available_tree_models) == 1:
            axes = [axes]
        
        for idx, name in enumerate(available_tree_models):
            model = results[name]['model']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:15]  # Top 15 features
                
                axes[idx].barh(range(len(indices)), importances[indices])
                axes[idx].set_yticks(range(len(indices)))
                axes[idx].set_yticklabels([self.feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{name} - Feature Importance')
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def cross_validate_with_time_series(self, X, y, n_splits=5):
        """Perform time series cross-validation"""
        print(f"Performing time series cross-validation with {n_splits} splits...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {}
        
        for name, result in self.models.items():
            model = result['model']
            scores = {'mae': [], 'rmse': [], 'r2': []}
            
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                if name in ['Linear Regression']:
                    X_train_cv = self.scaler.fit_transform(X_train_cv)
                    X_val_cv = self.scaler.transform(X_val_cv)
                
                # Create a new model instance for each fold
                if name == 'Linear Regression':
                    fold_model = LinearRegression()
                elif name == 'Random Forest':
                    fold_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                elif name == 'XGBoost' and HAS_XGB:
                    fold_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
                elif name == 'LightGBM' and HAS_LGB:
                    fold_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
                
                fold_model.fit(X_train_cv, y_train_cv)
                y_pred_cv = fold_model.predict(X_val_cv)
                
                scores['mae'].append(mean_absolute_error(y_val_cv, y_pred_cv))
                scores['rmse'].append(np.sqrt(mean_squared_error(y_val_cv, y_pred_cv)))
                scores['r2'].append(r2_score(y_val_cv, y_pred_cv))
            
            cv_results[name] = {
                'mae_mean': np.mean(scores['mae']),
                'mae_std': np.std(scores['mae']),
                'rmse_mean': np.mean(scores['rmse']),
                'rmse_std': np.std(scores['rmse']),
                'r2_mean': np.mean(scores['r2']),
                'r2_std': np.std(scores['r2'])
            }
            
            print(f"{name} CV Results:")
            print(f"  MAE: {cv_results[name]['mae_mean']:.2f} ± {cv_results[name]['mae_std']:.2f}")
            print(f"  RMSE: {cv_results[name]['rmse_mean']:.2f} ± {cv_results[name]['rmse_std']:.2f}")
            print(f"  R²: {cv_results[name]['r2_mean']:.4f} ± {cv_results[name]['r2_std']:.4f}")
            print()
        
        return cv_results
    
    def run_complete_analysis(self):
        """Run the complete sales forecasting analysis"""
        print("=== Sales Forecasting Analysis ===")
        print()
        
        # Generate or load data
        df = self.generate_sample_data(2000)
        print(f"Dataset shape: {df.shape}")
        print()
        
        # Create time features
        df_with_features = self.create_time_features(df)
        
        # Seasonal decomposition
        self.seasonal_decomposition(df_with_features)
        
        # Prepare features
        X, y, df_clean = self.prepare_features(df_with_features)
        
        # Train models
        X_test, y_test, results = self.train_models(X, y)
        
        # Plot results
        self.plot_results(X_test, y_test, results, df_with_features)
        
        # Cross-validation
        cv_results = self.cross_validate_with_time_series(X, y)
        
        # Summary
        print("=== SUMMARY ===")
        print("Best performing model based on R² score:")
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"Model: {best_model[0]}")
        print(f"R² Score: {best_model[1]['r2']:.4f}")
        print(f"RMSE: {best_model[1]['rmse']:.2f}")
        print(f"MAE: {best_model[1]['mae']:.2f}")

if __name__ == "__main__":
    # Run the complete analysis
    forecaster = SalesForecaster()
    forecaster.run_complete_analysis()
