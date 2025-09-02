# API Reference

## src.sales_forecasting

### SalesForecaster

The main class for basic sales forecasting functionality.

#### Methods

##### `__init__()`
Initialize the SalesForecaster with default parameters.

##### `generate_sample_data(n_samples=2000)`
Generate sample Walmart-like sales data for testing.

**Parameters:**
- `n_samples` (int): Number of samples to generate

**Returns:**
- `pd.DataFrame`: Generated sample data

##### `create_time_features(df)`
Create time-based features from the dataset.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe with Date column

**Returns:**
- `pd.DataFrame`: Dataframe with additional time features

##### `prepare_features(df)`
Prepare features for modeling by handling missing values and selecting appropriate columns.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe with features

**Returns:**
- `tuple`: (X, y, df_clean) where X is features, y is target, df_clean is cleaned dataframe

##### `train_models(X, y)`
Train multiple regression models on the prepared data.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable

**Returns:**
- `tuple`: (X_test, y_test, results) where results contains model performance metrics

##### `run_complete_analysis()`
Run the complete analysis pipeline from data generation to model evaluation.

## src.advanced_forecasting

### AdvancedSalesForecaster

Advanced class with bonus features including rolling averages, seasonal decomposition, and advanced models.

#### Methods

##### `load_data()`
Load Walmart sales data from CSV file or generate sample data.

**Returns:**
- `pd.DataFrame`: Loaded dataset

##### `create_advanced_features(df)`
Create comprehensive feature set including rolling averages and advanced time features.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

**Returns:**
- `pd.DataFrame`: Dataframe with advanced features

##### `seasonal_decomposition_advanced(df)`
Perform advanced seasonal decomposition using STL method.

**Parameters:**
- `df` (pd.DataFrame): Input dataframe

##### `train_advanced_models(X, y)`
Train advanced models including XGBoost and LightGBM.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable

**Returns:**
- `tuple`: (X_test, y_test, results) with model performance

##### `time_series_cross_validation(X, y, n_splits=5)`
Perform time series cross-validation.

**Parameters:**
- `X` (pd.DataFrame): Feature matrix
- `y` (pd.Series): Target variable
- `n_splits` (int): Number of CV splits

**Returns:**
- `dict`: Cross-validation results for each model

## src.data_downloader

### Functions

##### `download_sample_walmart_data()`
Generate realistic Walmart sales sample data with seasonal patterns.

**Returns:**
- `pd.DataFrame`: Generated sample dataset

##### `load_walmart_data()`
Load Walmart data from file or generate sample data if not found.

**Returns:**
- `pd.DataFrame`: Loaded or generated dataset

## Usage Examples

### Basic Usage

```python
from src.sales_forecasting import SalesForecaster

# Initialize forecaster
forecaster = SalesForecaster()

# Run complete analysis
forecaster.run_complete_analysis()
```

### Advanced Usage

```python
from src.advanced_forecasting import AdvancedSalesForecaster

# Initialize advanced forecaster
forecaster = AdvancedSalesForecaster()

# Run advanced analysis with all bonus features
forecaster.run_complete_advanced_analysis()
```

### Custom Data Loading

```python
from src.data_downloader import load_walmart_data

# Load data
df = load_walmart_data()

# Use with forecaster
forecaster = SalesForecaster()
df_features = forecaster.create_time_features(df)
X, y, df_clean = forecaster.prepare_features(df_features)
X_test, y_test, results = forecaster.train_models(X, y)
```
