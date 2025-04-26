import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')

class TimeGPTModel:
    """
    TimeGPT Model wrapper for time series forecasting
    
    Note: This requires the nixtla/timegpt package to be installed:
    pip install nixtla
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        freq: Optional[str] = None,
        finetune_steps: int = 0,
        finetune_loss: str = 'default',
        level: Optional[List[float]] = None
    ):
        """
        Initialize TimeGPT model
        
        Args:
            api_key (Optional[str]): TimeGPT API key (if None, tries to use environment variable)
            freq (Optional[str]): Time series frequency ('auto' for automatic detection)
            finetune_steps (int): Number of fine-tuning steps
            finetune_loss (str): Loss function for fine-tuning
            level (Optional[List[float]]): Confidence levels for prediction intervals
        """
        self.api_key = api_key
        self.freq = freq
        self.finetune_steps = finetune_steps
        self.finetune_loss = finetune_loss
        self.level = level if level is not None else [90]
        self.model = None
        self.fitted = False
        self.train_data = None
    
    def _initialize_model(self):
        """
        Initialize TimeGPT model
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Import TimeGPT (which requires API key)
            try:
                from nixtla import TimeGPT
            except ImportError:
                logger.error("TimeGPT package not found. Please install with: pip install nixtla")
                return False
            
            # Initialize TimeGPT model
            self.model = TimeGPT(api_key=self.api_key)
            return True
            
        except Exception as e:
            logger.error(f"Error initializing TimeGPT model: {str(e)}")
            return False
    
    def fit(
        self, 
        series: pd.Series,
        finetune: bool = True
    ) -> Dict[str, Any]:
        """
        Fit TimeGPT model to time series data
        
        Args:
            series (pd.Series): Time series data
            finetune (bool): Whether to fine-tune the model
            
        Returns:
            Dict[str, Any]: Dictionary with fit results
        """
        try:
            # Initialize model if not already done
            if self.model is None:
                success = self._initialize_model()
                if not success:
                    return {'fitted': False, 'error': "Failed to initialize TimeGPT model"}
            
            # Save training data
            self.train_data = series.copy()
            
            # Ensure index is datetime
            if not isinstance(series.index, pd.DatetimeIndex):
                logger.error("TimeGPT requires a DatetimeIndex. Please convert your data.")
                return {'fitted': False, 'error': "TimeGPT requires a DatetimeIndex"}
            
            # Determine frequency if not provided
            if self.freq is None or self.freq == 'auto':
                self.freq = pd.infer_freq(series.index)
                if self.freq is None:
                    # Try to infer from last few observations
                    self.freq = pd.infer_freq(series.index[-5:])
                    
                    if self.freq is None:
                        logger.warning("Could not infer frequency. Using 'D' (daily) as default.")
                        self.freq = 'D'
            
            # Convert to DataFrame in TimeGPT expected format
            df = pd.DataFrame({
                'unique_id': 'series',
                'ds': series.index,
                'y': series.values
            })
            
            # Fine-tune model if requested
            if finetune and self.finetune_steps > 0:
                # Import fine-tuning module
                try:
                    from nixtla.utils.finetune import tune_model
                except ImportError:
                    logger.warning("Fine-tuning module not found. Skipping fine-tuning.")
                    finetune = False
            
            if finetune and self.finetune_steps > 0:
                try:
                    logger.info(f"Fine-tuning TimeGPT model with {self.finetune_steps} steps")
                    tune_model(
                        self.model,
                        df=df,
                        finetune_steps=self.finetune_steps,
                        finetune_loss=self.finetune_loss
                    )
                except Exception as e:
                    logger.warning(f"Error during fine-tuning: {str(e)}. Continuing without fine-tuning.")
            
            self.fitted = True
            
            return {
                'fitted': True,
                'frequency': self.freq,
                'fine_tuned': finetune and self.finetune_steps > 0
            }
            
        except Exception as e:
            logger.error(f"Error fitting TimeGPT model: {str(e)}")
            self.fitted = False
            return {
                'fitted': False,
                'error': str(e)
            }
    
    def predict(
        self, 
        steps: int = 30,
        return_conf_int: bool = True
    ) -> Dict[str, Any]:
        """
        Generate forecasts from TimeGPT model
        
        Args:
            steps (int): Number of steps to forecast
            return_conf_int (bool): Whether to return confidence intervals
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return {'error': "Model not fitted. Call fit() first."}
        
        try:
            # Convert to DataFrame in TimeGPT expected format
            df = pd.DataFrame({
                'unique_id': 'series',
                'ds': self.train_data.index,
                'y': self.train_data.values
            })
            
            # Generate forecasts
            logger.info(f"Generating {steps} step forecast with TimeGPT")
            forecast = self.model.forecast(
                df=df,
                h=steps,
                freq=self.freq,
                level=self.level if return_conf_int else None
            )
            
            # Extract forecasts
            point_forecast = forecast.filter(like='TimeGPT').iloc[:, 0]
            
            # Create date index
            last_date = self.train_data.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq=self.freq)[1:]
            
            # Create Series with forecasts
            forecast_series = pd.Series(point_forecast.values, index=forecast_dates)
            
            # Return results
            result = {
                'forecast': forecast_series,
                'steps': steps
            }
            
            # Add confidence intervals if requested
            if return_conf_int:
                lower_cols = [col for col in forecast.columns if 'lower' in col]
                upper_cols = [col for col in forecast.columns if 'upper' in col]
                
                if lower_cols and upper_cols:
                    # Get the first confidence interval
                    lower_bound = pd.Series(forecast[lower_cols[0]].values, index=forecast_dates)
                    upper_bound = pd.Series(forecast[upper_cols[0]].values, index=forecast_dates)
                    
                    result['lower_bound'] = lower_bound
                    result['upper_bound'] = upper_bound
                    result['conf_level'] = self.level[0] / 100  # Convert to proportion
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return {'error': str(e)}
    
    def evaluate(self, test_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        
        Args:
            test_data (pd.Series): Test data
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return {'error': "Model not fitted. Call fit() first."}
        
        try:
            # Get forecast for test period
            forecast_result = self.predict(steps=len(test_data), return_conf_int=False)
            
            if 'error' in forecast_result:
                return forecast_result
            
            # Get forecasts
            forecasts = forecast_result['forecast']
            
            # Ensure indices match
            if isinstance(forecasts.index, pd.DatetimeIndex) and isinstance(test_data.index, pd.DatetimeIndex):
                # Align by date
                common_dates = forecasts.index.intersection(test_data.index)
                actuals = test_data.loc[common_dates].values
                preds = forecasts.loc[common_dates].values
            else:
                # Just use the first len(test_data) predictions
                actuals = test_data.values[:len(forecasts)]
                preds = forecasts.values[:len(actuals)]
            
            # Calculate error metrics
            mse = mean_squared_error(actuals, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals, preds)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
            
            # Calculate R-squared (coefficient of determination)
            r2 = r2_score(actuals, preds)
            
            # Return evaluation metrics
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {'error': str(e)}


def create_timegpt_model(
    series: pd.Series,
    api_key: Optional[str] = None,
    freq: Optional[str] = None,
    finetune_steps: int = 0,
    finetune_loss: str = 'default',
    level: Optional[List[float]] = None,
    finetune: bool = True
) -> Dict[str, Any]:
    """
    Create and fit TimeGPT model
    
    Args:
        series (pd.Series): Time series data
        api_key (Optional[str]): TimeGPT API key
        freq (Optional[str]): Time series frequency
        finetune_steps (int): Number of fine-tuning steps
        finetune_loss (str): Loss function for fine-tuning
        level (Optional[List[float]]): Confidence levels for prediction intervals
        finetune (bool): Whether to fine-tune the model
        
    Returns:
        Dict[str, Any]: Dictionary with model and fit results
    """
    try:
        # Initialize TimeGPT model
        model = TimeGPTModel(
            api_key=api_key,
            freq=freq,
            finetune_steps=finetune_steps,
            finetune_loss=finetune_loss,
            level=level
        )
        
        # Fit model
        fit_result = model.fit(series, finetune=finetune)
        
        # Return model and results
        return {
            'model': model,
            'fit_result': fit_result
        }
        
    except Exception as e:
        logger.error(f"Error creating TimeGPT model: {str(e)}")
        return {'error': str(e)}