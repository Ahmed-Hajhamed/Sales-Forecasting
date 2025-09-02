"""
Advanced Sales Forecasting with Bonus Features
Includes rolling averages, seasonal decomposition, XGBoost/LightGBM with time-aware validation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedSalesForecaster:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load the Walmart sales data"""
        try:
            # Try to load real Kaggle data first
            if os.path.exists('train.csv'):
                print("Loading Kaggle Walmart dataset...")
                df = pd.read_csv('train.csv')
                df['Date'] = pd.to_datetime(df['Date'])
            elif os.path.exists('walmart_sales_data.csv'):
                # Load sample data
                print("Loading sample dataset...")
                df = pd.read_csv('walmart_sales_data.csv')
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                print("No dataset found. Creating sample data...")
                from data_downloader import download_sample_walmart_data
                df = download_sample_walmart_data()
                df.to_csv('walmart_sales_data.csv', index=False)
                
            print(f"Loaded dataset with {len(df)} records")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def exploratory_data_analysis(self, df):
        """Perform comprehensive EDA"""
        print("=== Exploratory Data Analysis ===")
        
        # Basic statistics
        print("\nDataset Overview:")
        print(df.info())
        print("\nSales Statistics:")
        print(df['Weekly_Sales'].describe())
        
        # Create EDA plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Sales distribution
        axes[0, 0].hist(df['Weekly_Sales'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Weekly Sales Distribution')
        axes[0, 0].set_xlabel('Weekly Sales')
        axes[0, 0].set_ylabel('Frequency')
        
        # Sales over time
        daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        axes[0, 1].plot(daily_sales['Date'], daily_sales['Weekly_Sales'])
        axes[0, 1].set_title('Total Sales Over Time')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Total Sales')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sales by store
        store_sales = df.groupby('Store')['Weekly_Sales'].mean().reset_index()
        axes[0, 2].bar(store_sales['Store'], store_sales['Weekly_Sales'])
        axes[0, 2].set_title('Average Sales by Store')
        axes[0, 2].set_xlabel('Store')
        axes[0, 2].set_ylabel('Average Weekly Sales')
        
        # Sales by department
        dept_sales = df.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False).head(10)
        axes[1, 0].bar(range(len(dept_sales)), dept_sales.values)
        axes[1, 0].set_title('Top 10 Departments by Average Sales')
        axes[1, 0].set_xlabel('Department Rank')
        axes[1, 0].set_ylabel('Average Weekly Sales')
        axes[1, 0].set_xticks(range(len(dept_sales)))
        axes[1, 0].set_xticklabels(dept_sales.index, rotation=45)
        
        # Holiday vs non-holiday sales
        holiday_sales = df.groupby('IsHoliday')['Weekly_Sales'].mean()
        axes[1, 1].bar(['Non-Holiday', 'Holiday'], holiday_sales.values)
        axes[1, 1].set_title('Holiday vs Non-Holiday Sales')
        axes[1, 1].set_ylabel('Average Weekly Sales')
        
        # Monthly sales pattern
        df['Month'] = df['Date'].dt.month
        monthly_sales = df.groupby('Month')['Weekly_Sales'].mean()
        axes[1, 2].plot(monthly_sales.index, monthly_sales.values, marker='o')
        axes[1, 2].set_title('Monthly Sales Pattern')
        axes[1, 2].set_xlabel('Month')
        axes[1, 2].set_ylabel('Average Weekly Sales')
        axes[1, 2].set_xticks(range(1, 13))
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df
    
    def create_advanced_features(self, df):
        """Create comprehensive feature set including rolling averages and advanced time features"""
        print("Creating advanced feature set...")
        
        df = df.copy()
        df = df.sort_values(['Store', 'Dept', 'Date']).reset_index(drop=True)
        
        # Basic time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        
        # Cyclical encoding for better ML performance
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Lag features (multiple time horizons)
        print("Creating lag features...")
        for lag in [1, 2, 3, 7, 14, 21, 28, 35, 42]:
            df[f'Sales_lag_{lag}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].shift(lag)
        
        # Rolling statistics (BONUS: Rolling averages)
        print("Creating rolling statistics...")
        for window in [3, 7, 14, 21, 28, 42, 56]:
            # Rolling mean
            df[f'Sales_rolling_mean_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            
            # Rolling standard deviation
            df[f'Sales_rolling_std_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
            
            # Rolling min/max
            df[f'Sales_rolling_min_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=window, min_periods=1).min())
            df[f'Sales_rolling_max_{window}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.rolling(window=window, min_periods=1).max())
        
        # Growth rates and momentum
        print("Creating growth and momentum features...")
        for period in [1, 7, 14, 28]:
            df[f'Sales_growth_{period}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.pct_change(period))
            df[f'Sales_diff_{period}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.diff(period))
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f'Sales_ema_{alpha}'] = df.groupby(['Store', 'Dept'])['Weekly_Sales'].transform(lambda x: x.ewm(alpha=alpha).mean())
        
        # Store and department aggregated features
        print("Creating aggregated features...")
        
        # Store-level features
        store_stats = df.groupby(['Store', 'Date'])['Weekly_Sales'].agg(['mean', 'std', 'min', 'max']).reset_index()
        store_stats.columns = ['Store', 'Date', 'Store_mean_sales', 'Store_std_sales', 'Store_min_sales', 'Store_max_sales']
        df = df.merge(store_stats, on=['Store', 'Date'], how='left')
        
        # Department-level features
        dept_stats = df.groupby(['Dept', 'Date'])['Weekly_Sales'].agg(['mean', 'std']).reset_index()
        dept_stats.columns = ['Dept', 'Date', 'Dept_mean_sales', 'Dept_std_sales']
        df = df.merge(dept_stats, on=['Dept', 'Date'], how='left')
        
        # Holiday interaction features
        if 'IsHoliday' in df.columns:
            df['Holiday_Store'] = df['IsHoliday'].astype(int) * df['Store']
            df['Holiday_Dept'] = df['IsHoliday'].astype(int) * df['Dept']
        
        # Economic indicators interaction (if available)
        if 'Temperature' in df.columns:
            df['Temp_squared'] = df['Temperature'] ** 2
            df['Temp_Month'] = df['Temperature'] * df['Month']
            
        print(f"Created {len(df.columns)} features total")
        return df
    
    def seasonal_decomposition_advanced(self, df):
        """Advanced seasonal decomposition with multiple stores and departments"""
        print("Performing advanced seasonal decomposition...")
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            print("Statsmodels not available. Skipping seasonal decomposition.")
            return
        
        # Select top stores and departments by sales volume
        top_combinations = df.groupby(['Store', 'Dept'])['Weekly_Sales'].sum().nlargest(6).index
        
        fig, axes = plt.subplots(len(top_combinations), 4, figsize=(20, 4*len(top_combinations)))
        if len(top_combinations) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (store, dept) in enumerate(top_combinations):
            subset = df[(df['Store'] == store) & (df['Dept'] == dept)].copy()
            subset = subset.sort_values('Date').set_index('Date')
            
            if len(subset) < 100:  # Need enough data
                continue
                
            # Resample to weekly frequency to handle missing dates
            weekly_sales = subset['Weekly_Sales'].resample('W').mean().fillna(method='forward')
            
            if len(weekly_sales) < 52:  # Need at least 1 year
                continue
            
            try:
                # STL decomposition (more robust than classical)
                stl = STL(weekly_sales, seasonal=13)  # Quarterly seasonality
                decomposition = stl.fit()
                
                # Plot components
                decomposition.observed.plot(ax=axes[idx, 0], title=f'Store {store}, Dept {dept} - Original')
                decomposition.trend.plot(ax=axes[idx, 1], title='Trend')
                decomposition.seasonal.plot(ax=axes[idx, 2], title='Seasonal')
                decomposition.resid.plot(ax=axes[idx, 3], title='Residual')
                
                for ax in axes[idx]:
                    ax.tick_params(axis='x', rotation=45)
                    
            except Exception as e:
                print(f"Could not decompose Store {store}, Dept {dept}: {e}")
                continue
        
        plt.tight_layout()
        plt.savefig('advanced_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_modeling_data(self, df):
        """Prepare data for modeling with proper handling of categorical variables"""
        print("Preparing data for modeling...")
        
        # Remove rows with too many NaN values (from lag features)
        threshold = len(df.columns) * 0.7  # Keep rows with at least 70% non-null values
        df_clean = df.dropna(thresh=threshold)
        
        if len(df_clean) < len(df) * 0.5:
            print("Warning: Many rows removed due to NaN values. Using simpler feature set.")
            # Use only basic features without heavy lag features
            feature_cols = [col for col in df.columns if not any(x in col for x in ['lag_', 'rolling_', 'ema_', 'growth_', 'diff_'])]
            df_clean = df[feature_cols].dropna()
        
        # Define feature columns (exclude target and identifier columns)
        exclude_cols = ['Weekly_Sales', 'Date']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        # Handle categorical variables
        categorical_cols = ['Store', 'Dept']
        for col in categorical_cols:
            if col in feature_cols:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le
        
        # Convert boolean columns to int
        bool_cols = df_clean.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df_clean[col] = df_clean[col].astype(int)
        
        X = df_clean[feature_cols]
        y = df_clean['Weekly_Sales']
        
        self.feature_names = feature_cols
        print(f"Final feature set: {len(feature_cols)} features")
        print(f"Training data: {len(X)} samples")
        
        return X, y, df_clean
    
    def train_advanced_models(self, X, y):
        """Train advanced models with time-aware validation (BONUS)"""
        print("Training advanced models with time-aware validation...")
        
        # Time-aware split (last 20% for testing)
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Initialize models
        models = {}
        
        # Linear models (need scaling)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        models['Linear Regression'] = LinearRegression()
        models['Ridge Regression'] = Ridge(alpha=1.0)
        models['Lasso Regression'] = Lasso(alpha=0.1)
        
        # Tree-based models (no scaling needed)
        models['Random Forest'] = RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Advanced models (BONUS: XGBoost and LightGBM)
        try:
            import xgboost as xgb
            models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        except ImportError:
            print("XGBoost not available")
        
        try:
            import lightgbm as lgb
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=-1
            )
        except ImportError:
            print("LightGBM not available")
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
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
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mape': mape
                }
                
                print(f"  MAE: {mae:.2f}")
                print(f"  RMSE: {rmse:.2f}")
                print(f"  R²: {r2:.4f}")
                print(f"  MAPE: {mape:.2f}%")
                print()
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        self.models = results
        return X_test, y_test, results
    
    def time_series_cross_validation(self, X, y, n_splits=5):
        """Perform time series cross-validation (BONUS: Time-aware validation)"""
        print(f"Performing time series cross-validation with {n_splits} splits...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = {}
        
        for name, result in self.models.items():
            if result is None:
                continue
                
            print(f"Cross-validating {name}...")
            scores = {'mae': [], 'rmse': [], 'r2': [], 'mape': []}
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create fresh model instance
                model = result['model']
                
                try:
                    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_cv)
                        X_val_scaled = scaler.transform(X_val_cv)
                        
                        model.fit(X_train_scaled, y_train_cv)
                        y_pred_cv = model.predict(X_val_scaled)
                    else:
                        model.fit(X_train_cv, y_train_cv)
                        y_pred_cv = model.predict(X_val_cv)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_val_cv, y_pred_cv)
                    rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                    r2 = r2_score(y_val_cv, y_pred_cv)
                    mape = np.mean(np.abs((y_val_cv - y_pred_cv) / y_val_cv)) * 100
                    
                    scores['mae'].append(mae)
                    scores['rmse'].append(rmse)
                    scores['r2'].append(r2)
                    scores['mape'].append(mape)
                    
                except Exception as e:
                    print(f"Error in fold {fold} for {name}: {e}")
                    continue
            
            if scores['mae']:  # If we have any successful folds
                cv_results[name] = {
                    'mae_mean': np.mean(scores['mae']),
                    'mae_std': np.std(scores['mae']),
                    'rmse_mean': np.mean(scores['rmse']),
                    'rmse_std': np.std(scores['rmse']),
                    'r2_mean': np.mean(scores['r2']),
                    'r2_std': np.std(scores['r2']),
                    'mape_mean': np.mean(scores['mape']),
                    'mape_std': np.std(scores['mape'])
                }
                
                print(f"  MAE: {cv_results[name]['mae_mean']:.2f} ± {cv_results[name]['mae_std']:.2f}")
                print(f"  RMSE: {cv_results[name]['rmse_mean']:.2f} ± {cv_results[name]['rmse_std']:.2f}")
                print(f"  R²: {cv_results[name]['r2_mean']:.4f} ± {cv_results[name]['r2_std']:.4f}")
                print(f"  MAPE: {cv_results[name]['mape_mean']:.2f}% ± {cv_results[name]['mape_std']:.2f}%")
                print()
        
        return cv_results
    
    def create_comprehensive_visualizations(self, X_test, y_test, results, df_clean):
        """Create comprehensive visualizations"""
        print("Creating comprehensive visualizations...")
        
        # Get test dates
        test_dates = df_clean.iloc[int(len(df_clean) * 0.8):]['Date'].values
        
        # Model performance comparison
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Model performance metrics
        model_names = list(results.keys())
        metrics = ['mae', 'rmse', 'r2', 'mape']
        metric_values = {metric: [results[name][metric] for name in model_names] for metric in metrics}
        
        # MAE comparison
        axes[0, 0].bar(model_names, metric_values['mae'])
        axes[0, 0].set_title('Mean Absolute Error by Model')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[0, 1].bar(model_names, metric_values['r2'])
        axes[0, 1].set_title('R² Score by Model')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Time series plots for best 2 models
        best_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:2]
        
        for idx, (name, result) in enumerate(best_models):
            y_pred = result['predictions']
            
            # Sample for readability (every 10th point)
            sample_idx = slice(None, None, 10)
            
            axes[1, idx].plot(test_dates[sample_idx], y_test.values[sample_idx], 
                             label='Actual', alpha=0.7, linewidth=2)
            axes[1, idx].plot(test_dates[sample_idx], y_pred[sample_idx], 
                             label='Predicted', alpha=0.7, linewidth=2)
            axes[1, idx].set_title(f'{name} - Time Series (R² = {result["r2"]:.4f})')
            axes[1, idx].set_xlabel('Date')
            axes[1, idx].set_ylabel('Sales')
            axes[1, idx].legend()
            axes[1, idx].tick_params(axis='x', rotation=45)
        
        # Prediction accuracy scatter plots
        for idx, (name, result) in enumerate(best_models):
            y_pred = result['predictions']
            
            axes[2, idx].scatter(y_test, y_pred, alpha=0.5, s=1)
            axes[2, idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[2, idx].set_xlabel('Actual Sales')
            axes[2, idx].set_ylabel('Predicted Sales')
            axes[2, idx].set_title(f'{name} - Prediction Accuracy')
            
            # Add trend line
            z = np.polyfit(y_test, y_pred, 1)
            p = np.poly1d(z)
            axes[2, idx].plot(y_test, p(y_test), "b--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Feature importance for tree-based models
        self.plot_feature_importance_advanced(results)
        
    def plot_feature_importance_advanced(self, results):
        """Plot feature importance for tree-based models"""
        tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
        available_models = [name for name in tree_models if name in results]
        
        if not available_models:
            return
            
        fig, axes = plt.subplots(1, len(available_models), figsize=(6*len(available_models), 8))
        if len(available_models) == 1:
            axes = [axes]
        
        for idx, name in enumerate(available_models):
            model = results[name]['model']
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:20]  # Top 20 features
                
                feature_names = [self.feature_names[i] for i in indices]
                importance_values = importances[indices]
                
                axes[idx].barh(range(len(indices)), importance_values)
                axes[idx].set_yticks(range(len(indices)))
                axes[idx].set_yticklabels(feature_names)
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{name} - Top 20 Features')
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('advanced_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_advanced_analysis(self):
        """Run the complete advanced sales forecasting analysis"""
        print("=== ADVANCED SALES FORECASTING ANALYSIS ===")
        print()
        
        # Load data
        df = self.load_data()
        if df is None:
            print("Failed to load data. Exiting.")
            return
        
        # EDA
        df = self.exploratory_data_analysis(df)
        
        # Create advanced features
        df_features = self.create_advanced_features(df)
        
        # Seasonal decomposition (BONUS)
        self.seasonal_decomposition_advanced(df_features)
        
        # Prepare modeling data
        X, y, df_clean = self.prepare_modeling_data(df_features)
        
        # Train advanced models (BONUS: XGBoost/LightGBM with time-aware validation)
        X_test, y_test, results = self.train_advanced_models(X, y)
        
        # Time series cross-validation (BONUS)
        cv_results = self.time_series_cross_validation(X, y)
        
        # Comprehensive visualizations
        self.create_comprehensive_visualizations(X_test, y_test, results, df_clean)
        
        # Final summary
        print("=== FINAL SUMMARY ===")
        best_model = max(results.items(), key=lambda x: x[1]['r2'])
        print(f"Best Model: {best_model[0]}")
        print(f"Test R² Score: {best_model[1]['r2']:.4f}")
        print(f"Test RMSE: {best_model[1]['rmse']:.2f}")
        print(f"Test MAE: {best_model[1]['mae']:.2f}")
        print(f"Test MAPE: {best_model[1]['mape']:.2f}%")
        print()
        print("Analysis complete! Check the generated plots for detailed insights.")

if __name__ == "__main__":
    import os
    forecaster = AdvancedSalesForecaster()
    forecaster.run_complete_advanced_analysis()
