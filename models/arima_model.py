import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from typing import Dict, Any, Tuple, List, Optional, Union
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARIMAModel:
    """
    ARIMA Model for time series forecasting
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: Optional[str] = None,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True
    ):
        """
        Initialize ARIMA model
        
        Args:
            order (Tuple[int, int, int]): ARIMA order (p, d, q)
            seasonal_order (Optional[Tuple[int, int, int, int]]): Seasonal order (P, D, Q, s)
            trend (Optional[str]): Trend component ('n', 'c', 't', 'ct')
            enforce_stationarity (bool): Whether to enforce stationarity
            enforce_invertibility (bool): Whether to enforce invertibility
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.result = None
        self.fitted = False
        self.train_data = None
        
    def fit(self, series: pd.Series, **fit_kwargs) -> Dict[str, Any]:
        """
        Fit ARIMA model to time series data
        
        Args:
            series (pd.Series): Time series data
            **fit_kwargs: Additional arguments to pass to SARIMAX.fit()
            
        Returns:
            Dict[str, Any]: Dictionary with fit results
        """
        try:
            # Save training data
            self.train_data = series.copy()
            
            # Create model
            if self.seasonal_order:
                self.model = SARIMAX(
                    series,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend=self.trend,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility
                )
                model_type = "SARIMA"
            else:
                self.model = SARIMAX(
                    series,
                    order=self.order,
                    trend=self.trend,
                    enforce_stationarity=self.enforce_stationarity,
                    enforce_invertibility=self.enforce_invertibility
                )
                model_type = "ARIMA"
            
            # Set default fit parameters if not provided
            if 'disp' not in fit_kwargs:
                fit_kwargs['disp'] = False
            
            # Fit model
            logger.info(f"Fitting {model_type}{self.order} model")
            self.result = self.model.fit(**fit_kwargs)
            self.fitted = True
            
            # Get model summary
            summary = self.result.summary()
            
            # Extract key metrics
            aic = self.result.aic
            bic = self.result.bic
            
            # Get residuals
            residuals = self.result.resid
            
            # Calculate residual statistics
            residual_mean = residuals.mean()
            residual_std = residuals.std()
            
            # Calculate model parameters
            params = self.result.params
            
            # Return fit results
            return {
                'fitted': True,
                'aic': aic,
                'bic': bic,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'params': params,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            self.fitted = False
            return {
                'fitted': False,
                'error': str(e)
            }
    
    def predict(
        self, 
        steps: int = 1,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Generate forecasts from fitted model
        
        Args:
            steps (int): Number of steps to forecast
            return_conf_int (bool): Whether to return confidence intervals
            alpha (float): Significance level for confidence intervals
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return {'error': "Model not fitted. Call fit() first."}
        
        try:
            # Generate forecast
            forecast = self.result.get_forecast(steps=steps)
            
            # Get predicted mean
            pred_mean = forecast.predicted_mean
            
            # Get confidence intervals if requested
            if return_conf_int:
                pred_conf = forecast.conf_int(alpha=alpha)
                lower_bound = pred_conf.iloc[:, 0]
                upper_bound = pred_conf.iloc[:, 1]
            else:
                lower_bound = None
                upper_bound = None
            
            # Get forecast dates (if index is datetime)
            if isinstance(self.train_data.index, pd.DatetimeIndex):
                # Calculate forecast dates
                last_date = self.train_data.index[-1]
                freq = pd.infer_freq(self.train_data.index)
                if freq is None:
                    # Try to infer frequency from last few observations
                    freq = pd.infer_freq(self.train_data.index[-5:])
                    
                if freq is not None:
                    forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
                    pred_mean.index = forecast_dates
                    if return_conf_int:
                        lower_bound.index = forecast_dates
                        upper_bound.index = forecast_dates
            
            # Return forecast results
            result = {
                'forecast': pred_mean,
                'steps': steps
            }
            
            if return_conf_int:
                result['lower_bound'] = lower_bound
                result['upper_bound'] = upper_bound
                result['conf_level'] = 1 - alpha
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return {'error': str(e)}
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostics
        
        Returns:
            Dict[str, Any]: Dictionary with diagnostic results
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return {'error': "Model not fitted. Call fit() first."}
        
        try:
            # Get residuals
            residuals = self.result.resid
            
            # Calculate residual statistics
            residual_mean = residuals.mean()
            residual_std = residuals.std()
            
            # Perform Ljung-Box test for autocorrelation in residuals
            lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
            
            # Check residuals for normality
            jb_test = sm.stats.jarque_bera(residuals)
            
            # Return diagnostic results
            return {
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'ljung_box_test': {
                    'statistic': lb_test['lb_stat'].values[0],
                    'p_value': lb_test['lb_pvalue'].values[0],
                    'no_autocorr': lb_test['lb_pvalue'].values[0] > 0.05
                },
                'jarque_bera_test': {
                    'statistic': jb_test[0],
                    'p_value': jb_test[1],
                    'is_normal': jb_test[1] > 0.05
                },
                'residuals': residuals
            }
            
        except Exception as e:
            logger.error(f"Error getting model diagnostics: {str(e)}")
            return {'error': str(e)}
    
    def get_in_sample_predictions(self) -> pd.Series:
        """
        Get in-sample predictions (fitted values)
        
        Returns:
            pd.Series: Series with in-sample predictions
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return pd.Series()
        
        try:
            # Get in-sample predictions
            predictions = self.result.get_prediction()
            predicted_mean = predictions.predicted_mean
            
            return predicted_mean
            
        except Exception as e:
            logger.error(f"Error getting in-sample predictions: {str(e)}")
            return pd.Series()
    
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
            # Generate forecasts for test period
            forecast_result = self.predict(steps=len(test_data), return_conf_int=False)
            
            if 'error' in forecast_result:
                return forecast_result
            
            # Get forecasts
            forecasts = forecast_result['forecast']
            
            # Ensure indices match
            if not isinstance(forecasts.index, pd.DatetimeIndex) or not isinstance(test_data.index, pd.DatetimeIndex):
                # If indices are not datetime, use values directly
                y_true = test_data.values
                y_pred = forecasts.values
            else:
                # Align by date
                common_dates = forecasts.index.intersection(test_data.index)
                y_true = test_data.loc[common_dates].values
                y_pred = forecasts.loc[common_dates].values
            
            # Calculate error metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Calculate R-squared (coefficient of determination)
            r2 = r2_score(y_true, y_pred)
            
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


def auto_arima_model(
    series: pd.Series,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    seasonal: bool = False,
    m: int = 1,
    information_criterion: str = 'aic',
    suppress_warnings: bool = True,
    return_arima_model: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Automatically find the best ARIMA model using auto_arima
    
    Args:
        series (pd.Series): Time series data
        max_p (int): Maximum p value
        max_d (int): Maximum d value
        max_q (int): Maximum q value
        seasonal (bool): Whether to fit seasonal ARIMA
        m (int): Seasonal period
        information_criterion (str): Information criterion to use ('aic', 'bic', 'hqic', 'oob')
        suppress_warnings (bool): Whether to suppress warnings
        return_arima_model (bool): Whether to return ARIMA model
        **kwargs: Additional arguments to pass to auto_arima
        
    Returns:
        Dict[str, Any]: Dictionary with best model and results
    """
    try:
        # Set default kwargs if not provided
        if 'trace' not in kwargs:
            kwargs['trace'] = True
        if 'error_action' not in kwargs:
            kwargs['error_action'] = 'ignore'
        if 'stepwise' not in kwargs:
            kwargs['stepwise'] = True
        
        # Fit auto_arima model
        logger.info("Fitting auto_arima model")
        auto_model = auto_arima(
            series,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            seasonal=seasonal,
            m=m,
            information_criterion=information_criterion,
            suppress_warnings=suppress_warnings,
            **kwargs
        )
        
        # Get best order
        best_order = auto_model.order
        
        # Get best seasonal order if applicable
        if seasonal:
            best_seasonal_order = auto_model.seasonal_order
        else:
            best_seasonal_order = None
        
        # Get model summary
        summary = auto_model.summary()
        
        # Get AIC and BIC
        aic = auto_model.aic()
        try:
            bic = auto_model.bic()
        except:
            bic = None
        
        # Create result dictionary
        result = {
            'best_order': best_order,
            'best_seasonal_order': best_seasonal_order,
            'aic': aic,
            'bic': bic,
            'summary': summary
        }
        
        if return_arima_model:
            result['model'] = auto_model
        
        # Create and fit ARIMA model with best parameters
        arima_model = ARIMAModel(
            order=best_order,
            seasonal_order=best_seasonal_order
        )
        fit_result = arima_model.fit(series)
        
        result['arima_model'] = arima_model
        result['fit_result'] = fit_result
        
        return result
        
    except Exception as e:
        logger.error(f"Error in auto_arima: {str(e)}")
        return {'error': str(e)}