import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional, Any, Union
from statsmodels.tsa.stattools import adfuller

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class to handle data preprocessing for time series analysis
    """
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Calculate daily returns
        
        Args:
            df (pd.DataFrame): DataFrame with time series data
            column (str): Column to calculate returns for
            
        Returns:
            pd.DataFrame: DataFrame with additional 'returns' column
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate percentage returns
        result_df['returns'] = result_df[column].pct_change() * 100
        
        # Calculate log returns
        result_df['log_returns'] = np.log(result_df[column] / result_df[column].shift(1)) * 100
        
        # Calculate cumulative returns
        result_df['cumulative_returns'] = (1 + result_df['returns'] / 100).cumprod() - 1
        
        return result_df
    
    @staticmethod
    def check_stationarity(series: pd.Series) -> Dict[str, Any]:
        """
        Check stationarity of a time series using the Augmented Dickey-Fuller test
        
        Args:
            series (pd.Series): Time series to check
            
        Returns:
            Dict[str, Any]: Dictionary with stationarity test results
        """
        # Drop NaN values
        series = series.dropna()
        
        if len(series) < 20:
            return {
                "stationary": False,
                "p_value": None,
                "message": "Insufficient data points for stationarity test (minimum 20 required)"
            }
        
        try:
            # Perform ADF test
            result = adfuller(series)
            
            # Extract results
            adf_statistic = result[0]
            p_value = result[1]
            critical_values = result[4]
            
            # Determine if stationary
            is_stationary = p_value < 0.05
            
            # Prepare detailed message
            if is_stationary:
                message = f"Series is stationary (p-value: {p_value:.4f})"
            else:
                message = f"Series is not stationary (p-value: {p_value:.4f})"
            
            # Return results
            return {
                "stationary": is_stationary,
                "p_value": p_value,
                "adf_statistic": adf_statistic,
                "critical_values": critical_values,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Error performing stationarity test: {str(e)}")
            return {
                "stationary": False,
                "p_value": None,
                "message": f"Error performing stationarity test: {str(e)}"
            }
    
    @staticmethod
    def make_stationary(series: pd.Series, method: str = 'diff') -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Transform a series to make it stationary
        
        Args:
            series (pd.Series): Time series to transform
            method (str): Method to use ('diff', 'log_diff', 'pct_change')
            
        Returns:
            Tuple[pd.Series, Dict[str, Any]]: (Transformed series, stationarity test results)
        """
        # Drop NaN values
        series = series.dropna()
        
        if method == 'diff':
            transformed = series.diff().dropna()
        elif method == 'log_diff':
            transformed = np.log(series).diff().dropna()
        elif method == 'pct_change':
            transformed = series.pct_change().dropna()
        else:
            raise ValueError(f"Invalid method: {method}. Must be one of: diff, log_diff, pct_change")
        
        # Check stationarity of transformed series
        stationarity_results = DataProcessor.check_stationarity(transformed)
        
        return transformed, stationarity_results
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize data using min-max scaling
        
        Args:
            df (pd.DataFrame): DataFrame to normalize
            columns (Optional[List[str]]): Columns to normalize (None for all numeric columns)
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Determine columns to normalize
        if columns is None:
            columns = result_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Normalize each column
        for col in columns:
            if col in result_df.columns:
                min_val = result_df[col].min()
                max_val = result_df[col].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    result_df[f'{col}_normalized'] = (result_df[col] - min_val) / (max_val - min_val)
                else:
                    result_df[f'{col}_normalized'] = 0
        
        return result_df
    
    @staticmethod
    def prepare_for_modeling(
        df: pd.DataFrame, 
        target_col: str = 'close', 
        sequence_length: int = 10,
        target_steps: int = 1,
        normalize: bool = True,
        test_size: float = 0.2,
        features: Optional[List[str]] = None
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Prepare data for time series modeling
        
        Args:
            df (pd.DataFrame): DataFrame with time series data
            target_col (str): Target column for prediction
            sequence_length (int): Number of time steps to use as input features
            target_steps (int): Number of time steps to predict ahead
            normalize (bool): Whether to normalize the data
            test_size (float): Proportion of data to use for testing
            features (Optional[List[str]]): List of feature columns to use (None for only target_col)
            
        Returns:
            Dict[str, Union[np.ndarray, pd.DataFrame]]: Dictionary with prepared data
        """
        # Create a copy to avoid modifying the original
        data = df.copy()
        
        # Determine features to use
        if features is None:
            features = [target_col]
        
        # Ensure all selected features exist in the dataframe
        for feature in features:
            if feature not in data.columns:
                raise ValueError(f"Feature '{feature}' not found in dataframe columns: {data.columns.tolist()}")
        
        # Extract features
        feature_data = data[features].copy()
        
        # Handle missing values
        feature_data = feature_data.dropna()
        
        if len(feature_data) < sequence_length + target_steps:
            raise ValueError(f"Not enough data points after removing NaN values. Have {len(feature_data)}, need at least {sequence_length + target_steps}")
        
        # Normalize if requested
        scaler = None
        if normalize:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            feature_data_normalized = pd.DataFrame(
                scaler.fit_transform(feature_data),
                columns=feature_data.columns,
                index=feature_data.index
            )
            data_to_use = feature_data_normalized
        else:
            data_to_use = feature_data
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(data_to_use) - sequence_length - target_steps + 1):
            # Input sequence
            X.append(data_to_use.values[i:(i + sequence_length)])
            
            # Target value(s) - either a single value or a sequence depending on target_steps
            if target_steps == 1:
                y.append(data_to_use[target_col].values[i + sequence_length])
            else:
                y.append(data_to_use[target_col].values[i + sequence_length:i + sequence_length + target_steps])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Get the corresponding dates
        dates = feature_data.index[sequence_length:].values
        train_dates = dates[:split_idx]
        test_dates = dates[split_idx:]
        
        # Return the prepared data
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'feature_names': features,
            'scaler': scaler,
            'original_data': feature_data
        }
    
    @staticmethod
    def inverse_transform_predictions(predictions: np.ndarray, scaler, target_idx: int = 0) -> np.ndarray:
        """
        Inverse transform normalized predictions back to original scale
        
        Args:
            predictions (np.ndarray): Normalized predictions
            scaler: Scaler used for normalization (e.g., MinMaxScaler)
            target_idx (int): Index of the target variable in the scaler's feature list
            
        Returns:
            np.ndarray: Predictions in original scale
        """
        if scaler is None:
            return predictions
        
        # Reshape predictions for inverse transform if needed
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Create a dummy array for inverse transform
        dummy = np.zeros((predictions.shape[0], scaler.scale_.shape[0]))
        dummy[:, target_idx] = predictions[:, 0] if predictions.shape[1] == 1 else predictions
        
        # Inverse transform
        inversed = scaler.inverse_transform(dummy)
        
        # Extract the target column
        return inversed[:, target_idx]