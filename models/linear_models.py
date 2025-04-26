import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, Lambda, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to set up GPU memory growth to avoid memory allocation errors
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU memory growth set for {len(gpus)} GPUs")
except Exception as e:
    logger.warning(f"Error setting GPU memory growth: {str(e)}")

class SeasonalDecompositionLayer(Layer):
    """
    Custom Keras layer for seasonal decomposition of time series
    """
    
    def __init__(self, seq_len: int, **kwargs):
        """
        Initialize layer
        
        Args:
            seq_len (int): Sequence length
        """
        super(SeasonalDecompositionLayer, self).__init__(**kwargs)
        self.seq_len = seq_len
    
    def build(self, input_shape):
        """
        Build layer
        
        Args:
            input_shape: Input shape
        """
        super(SeasonalDecompositionLayer, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass
        
        Args:
            inputs: Layer inputs
            
        Returns:
            Tuple: (trend, seasonality)
        """
        # Calculate mean across sequence dimension for trend
        trend = tf.reduce_mean(inputs, axis=1, keepdims=True)
        trend = tf.tile(trend, [1, self.seq_len, 1])
        
        # Seasonality is input - trend
        seasonality = inputs - trend
        
        return trend, seasonality
    
    def compute_output_shape(self, input_shape):
        """
        Compute output shape
        
        Args:
            input_shape: Input shape
            
        Returns:
            Tuple: Output shapes for trend and seasonality
        """
        return [(input_shape[0], 1, input_shape[2]), input_shape]
    
    def get_config(self):
        """
        Get layer configuration
        
        Returns:
            Dict: Layer configuration
        """
        config = super(SeasonalDecompositionLayer, self).get_config()
        config.update({"seq_len": self.seq_len})
        return config


class DLinearModel:
    """
    D-Linear Model for time series forecasting
    
    This model decomposes time series into trend and seasonal components
    and applies separate linear layers to each component.
    """
    
    def __init__(
        self,
        sequence_length: int = 24,
        horizon: int = 12,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        validation_split: float = 0.2
    ):
        """
        Initialize D-Linear model
        
        Args:
            sequence_length (int): Input sequence length
            horizon (int): Forecast horizon
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs for training
            patience (int): Patience for early stopping
            validation_split (float): Fraction of data to use for validation
        """
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        self.fitted = False
        self.train_data = None
    
    def _build_model(self) -> Model:
        """
        Build D-Linear model
        
        Returns:
            Model: Keras Model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, 1))
        
        # Decomposition layer
        trend, seasonality = SeasonalDecompositionLayer(self.sequence_length)(inputs)
        
        # Flatten inputs for linear layers
        trend_flat = Flatten()(trend)
        seasonality_flat = Flatten()(seasonality)
        
        # Linear layers for trend and seasonality
        trend_output = Dense(self.horizon)(trend_flat)
        seasonality_output = Dense(self.horizon)(seasonality_flat)
        
        # Combine outputs
        outputs = tf.keras.layers.Add()([trend_output, seasonality_output])
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
    def _create_sequences(
        self, 
        data: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences
        
        Args:
            data (np.ndarray): Time series data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (input sequences) and y (target sequences)
        """
        X, y = [], []
        
        # Create sliding window sequences
        for i in range(len(data) - self.sequence_length - self.horizon + 1):
            # Input sequence
            X.append(data[i:(i + self.sequence_length)])
            
            # Target sequence
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.horizon)])
        
        return np.array(X), np.array(y)
    
    def fit(
        self, 
        series: pd.Series, 
        validation_data: Optional[pd.Series] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Fit model to time series data
        
        Args:
            series (pd.Series): Time series data
            validation_data (Optional[pd.Series]): Validation data
            verbose (int): Verbosity level for training
            
        Returns:
            Dict[str, Any]: Dictionary with fit results
        """
        try:
            # Save training data
            self.train_data = series.copy()
            
            # Scale data
            data = series.values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            # Reshape X for model input [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Prepare validation data if provided
            if validation_data is not None:
                val_data = validation_data.values.reshape(-1, 1)
                scaled_val_data = self.scaler.transform(val_data)
                X_val, y_val = self._create_sequences(scaled_val_data)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                validation = (X_val, y_val)
                internal_validation_split = 0  # Don't use internal validation split
            else:
                validation = None
                internal_validation_split = self.validation_split
            
            # Build model
            self.model = self._build_model()
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.patience // 2,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            logger.info(f"Training D-Linear model with {len(X)} sequences")
            self.history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=internal_validation_split,
                validation_data=validation,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.fitted = True
            
            # Calculate training metrics
            train_pred = self.model.predict(X, verbose=0)
            train_pred_reshaped = train_pred.reshape(-1, 1)
            train_y_reshaped = y.reshape(-1, 1)
            
            # Convert predictions back to original scale
            train_pred_original = self.scaler.inverse_transform(train_pred_reshaped)
            train_y_original = self.scaler.inverse_transform(train_y_reshaped)
            
            train_rmse = np.sqrt(mean_squared_error(train_y_original, train_pred_original))
            train_mae = mean_absolute_error(train_y_original, train_pred_original)
            
            # Return fit results
            return {
                'fitted': True,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'epochs_trained': len(self.history.history['loss']),
                'final_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history['val_loss'][-1] if 'val_loss' in self.history.history else None,
                'history': self.history.history
            }
            
        except Exception as e:
            logger.error(f"Error fitting D-Linear model: {str(e)}")
            self.fitted = False
            return {
                'fitted': False,
                'error': str(e)
            }
    
    def predict(self, steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate forecasts from fitted model
        
        Args:
            steps (Optional[int]): Number of steps to forecast (default: self.horizon)
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return {'error': "Model not fitted. Call fit() first."}
        
        try:
            # Determine forecast horizon
            if steps is None:
                steps = self.horizon
            
            # Create multi-step forecast
            if steps <= self.horizon:
                # We can forecast directly
                return self._direct_forecast(steps)
            else:
                # Need to do iterative forecasting
                return self._iterative_forecast(steps)
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return {'error': str(e)}
    
    def _direct_forecast(self, steps: int) -> Dict[str, Any]:
        """
        Generate direct forecast (no more than self.horizon steps)
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        # Get the latest data for input
        last_data = self.train_data[-self.sequence_length:].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.transform(last_data)
        
        # Reshape for model input
        input_data = scaled_data.reshape(1, self.sequence_length, 1)
        
        # Generate forecast
        forecast_scaled = self.model.predict(input_data, verbose=0)[0]
        
        # Take only the requested number of steps
        forecast_scaled = forecast_scaled[:steps].reshape(-1, 1)
        
        # Scale back to original scale
        forecast = self.scaler.inverse_transform(forecast_scaled).flatten()
        
        # Create date index for forecasts if original data has DatetimeIndex
        if isinstance(self.train_data.index, pd.DatetimeIndex):
            # Calculate forecast dates
            last_date = self.train_data.index[-1]
            freq = pd.infer_freq(self.train_data.index)
            if freq is None:
                # Try to infer frequency from last few observations
                freq = pd.infer_freq(self.train_data.index[-5:])
                
            if freq is not None:
                forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
                forecast = pd.Series(forecast, index=forecast_dates)
            else:
                forecast = pd.Series(forecast)
        else:
            forecast = pd.Series(forecast)
        
        return {
            'forecast': forecast,
            'steps': steps
        }
    
    def _iterative_forecast(self, steps: int) -> Dict[str, Any]:
        """
        Generate iterative forecast for longer horizons
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        # Get initial input data
        initial_data = self.train_data[-self.sequence_length:].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(initial_data)
        
        # Initialize arrays for forecasts
        all_forecasts = []
        current_input = scaled_data.copy()
        
        # Generate forecasts iteratively
        remaining_steps = steps
        
        while remaining_steps > 0:
            # Prepare input for current iteration
            input_data = current_input[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Generate forecast for current horizon
            current_horizon = min(self.horizon, remaining_steps)
            forecast_scaled = self.model.predict(input_data, verbose=0)[0][:current_horizon]
            
            # Add to forecasts
            all_forecasts.append(forecast_scaled)
            
            # Update input data for next iteration
            current_input = np.vstack([current_input, forecast_scaled.reshape(-1, 1)])
            
            # Update remaining steps
            remaining_steps -= current_horizon
        
        # Combine all forecasts
        forecast_scaled = np.concatenate(all_forecasts).reshape(-1, 1)
        
        # Scale back to original scale
        forecast = self.scaler.inverse_transform(forecast_scaled).flatten()
        
        # Create date index for forecasts if original data has DatetimeIndex
        if isinstance(self.train_data.index, pd.DatetimeIndex):
            # Calculate forecast dates
            last_date = self.train_data.index[-1]
            freq = pd.infer_freq(self.train_data.index)
            if freq is None:
                # Try to infer frequency from last few observations
                freq = pd.infer_freq(self.train_data.index[-5:])
                
            if freq is not None:
                forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
                forecast = pd.Series(forecast, index=forecast_dates)
            else:
                forecast = pd.Series(forecast)
        else:
            forecast = pd.Series(forecast)
        
        return {
            'forecast': forecast,
            'steps': steps
        }
    
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
            forecast_result = self.predict(steps=len(test_data))
            
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


class NLinearModel:
    """
    N-Linear Model for time series forecasting
    
    This model applies normalization before the linear layer.
    """
    
    def __init__(
        self,
        sequence_length: int = 24,
        horizon: int = 12,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        validation_split: float = 0.2
    ):
        """
        Initialize N-Linear model
        
        Args:
            sequence_length (int): Input sequence length
            horizon (int): Forecast horizon
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs for training
            patience (int): Patience for early stopping
            validation_split (float): Fraction of data to use for validation
        """
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.validation_split = validation_split
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        self.fitted = False
        self.train_data = None
    
    def _build_model(self) -> Model:
        """
        Build N-Linear model
        
        Returns:
            Model: Keras Model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, 1))
        
        # Calculate mean and std for normalization
        mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(inputs)
        mean = Lambda(lambda x: tf.tile(x, [1, self.sequence_length, 1]))(mean)
        
        std = Lambda(lambda x: tf.math.reduce_std(x, axis=1, keepdims=True))(inputs)
        std = Lambda(lambda x: tf.tile(x, [1, self.sequence_length, 1]))(std)
        
        # Normalize input
        normalized = Lambda(lambda x: (x[0] - x[1]) / (x[2] + 1e-8))([inputs, mean, std])
        
        # Flatten normalized input
        flattened = Flatten()(normalized)
        
        # Linear layer
        output = Dense(self.horizon)(flattened)
        
        # Denormalize output
        # We extract the last value of mean and std for denormalization
        last_mean = Lambda(lambda x: x[:, -1:, :])(mean)
        last_mean = Flatten()(last_mean)
        
        last_std = Lambda(lambda x: x[:, -1:, :])(std)
        last_std = Flatten()(last_std)
        
        # Broadcast to match output shape
        last_mean = Lambda(lambda x: tf.tile(x[:, tf.newaxis], [1, self.horizon]))(last_mean)
        last_std = Lambda(lambda x: tf.tile(x[:, tf.newaxis], [1, self.horizon]))(last_std)
        
        # Denormalize
        denormalized = Lambda(lambda x: x[0] * x[2] + x[1])([output, last_mean, last_std])
        
        # Create model
        model = Model(inputs=inputs, outputs=denormalized)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
    def _create_sequences(
        self, 
        data: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences
        
        Args:
            data (np.ndarray): Time series data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (input sequences) and y (target sequences)
        """
        X, y = [], []
        
        # Create sliding window sequences
        for i in range(len(data) - self.sequence_length - self.horizon + 1):
            # Input sequence
            X.append(data[i:(i + self.sequence_length)])
            
            # Target sequence
            y.append(data[(i + self.sequence_length):(i + self.sequence_length + self.horizon)])
        
        return np.array(X), np.array(y)
    
    def fit(
        self, 
        series: pd.Series, 
        validation_data: Optional[pd.Series] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Fit model to time series data
        
        Args:
            series (pd.Series): Time series data
            validation_data (Optional[pd.Series]): Validation data
            verbose (int): Verbosity level for training
            
        Returns:
            Dict[str, Any]: Dictionary with fit results
        """
        try:
            # Save training data
            self.train_data = series.copy()
            
            # Scale data
            data = series.values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            
            # Reshape X for model input [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Prepare validation data if provided
            if validation_data is not None:
                val_data = validation_data.values.reshape(-1, 1)
                scaled_val_data = self.scaler.transform(val_data)
                X_val, y_val = self._create_sequences(scaled_val_data)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                validation = (X_val, y_val)
                internal_validation_split = 0  # Don't use internal validation split
            else:
                validation = None
                internal_validation_split = self.validation_split
            
            # Build model
            self.model = self._build_model()
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.patience // 2,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            logger.info(f"Training N-Linear model with {len(X)} sequences")
            self.history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=internal_validation_split,
                validation_data=validation,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.fitted = True
            
            # Calculate training metrics
            train_pred = self.model.predict(X, verbose=0)
            train_pred_reshaped = train_pred.reshape(-1, 1)
            train_y_reshaped = y.reshape(-1, 1)
            
            # Convert predictions back to original scale
            train_pred_original = self.scaler.inverse_transform(train_pred_reshaped)
            train_y_original = self.scaler.inverse_transform(train_y_reshaped)
            
            train_rmse = np.sqrt(mean_squared_error(train_y_original, train_pred_original))
            train_mae = mean_absolute_error(train_y_original, train_pred_original)
            
            # Return fit results
            return {
                'fitted': True,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'epochs_trained': len(self.history.history['loss']),
                'final_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history['val_loss'][-1] if 'val_loss' in self.history.history else None,
                'history': self.history.history
            }
            
        except Exception as e:
            logger.error(f"Error fitting N-Linear model: {str(e)}")
            self.fitted = False
            return {
                'fitted': False,
                'error': str(e)
            }
    
    def predict(self, steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate forecasts from fitted model
        
        Args:
            steps (Optional[int]): Number of steps to forecast (default: self.horizon)
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return {'error': "Model not fitted. Call fit() first."}
        
        try:
            # Determine forecast horizon
            if steps is None:
                steps = self.horizon
            
            # Create multi-step forecast
            if steps <= self.horizon:
                # We can forecast directly
                return self._direct_forecast(steps)
            else:
                # Need to do iterative forecasting
                return self._iterative_forecast(steps)
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            return {'error': str(e)}
    
    def _direct_forecast(self, steps: int) -> Dict[str, Any]:
        """
        Generate direct forecast (no more than self.horizon steps)
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        # Get the latest data for input
        last_data = self.train_data[-self.sequence_length:].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.transform(last_data)
        
        # Reshape for model input
        input_data = scaled_data.reshape(1, self.sequence_length, 1)
        
        # Generate forecast
        forecast_scaled = self.model.predict(input_data, verbose=0)[0]
        
        # Take only the requested number of steps
        forecast_scaled = forecast_scaled[:steps].reshape(-1, 1)
        
        # Scale back to original scale
        forecast = self.scaler.inverse_transform(forecast_scaled).flatten()
        
        # Create date index for forecasts if original data has DatetimeIndex
        if isinstance(self.train_data.index, pd.DatetimeIndex):
            # Calculate forecast dates
            last_date = self.train_data.index[-1]
            freq = pd.infer_freq(self.train_data.index)
            if freq is None:
                # Try to infer frequency from last few observations
                freq = pd.infer_freq(self.train_data.index[-5:])
                
            if freq is not None:
                forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
                forecast = pd.Series(forecast, index=forecast_dates)
            else:
                forecast = pd.Series(forecast)
        else:
            forecast = pd.Series(forecast)
        
        return {
            'forecast': forecast,
            'steps': steps
        }
    
    def _iterative_forecast(self, steps: int) -> Dict[str, Any]:
        """
        Generate iterative forecast for longer horizons
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        # Get initial input data
        initial_data = self.train_data[-self.sequence_length:].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(initial_data)
        
        # Initialize arrays for forecasts
        all_forecasts = []
        current_input = scaled_data.copy()
        
        # Generate forecasts iteratively
        remaining_steps = steps
        
        while remaining_steps > 0:
            # Prepare input for current iteration
            input_data = current_input[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Generate forecast for current horizon
            current_horizon = min(self.horizon, remaining_steps)
            forecast_scaled = self.model.predict(input_data, verbose=0)[0][:current_horizon]
            
            # Add to forecasts
            all_forecasts.append(forecast_scaled)
            
            # Update input data for next iteration
            current_input = np.vstack([current_input, forecast_scaled.reshape(-1, 1)])
            
            # Update remaining steps
            remaining_steps -= current_horizon
        
        # Combine all forecasts
        forecast_scaled = np.concatenate(all_forecasts).reshape(-1, 1)
        
        # Scale back to original scale
        forecast = self.scaler.inverse_transform(forecast_scaled).flatten()
        
        # Create date index for forecasts if original data has DatetimeIndex
        if isinstance(self.train_data.index, pd.DatetimeIndex):
            # Calculate forecast dates
            last_date = self.train_data.index[-1]
            freq = pd.infer_freq(self.train_data.index)
            if freq is None:
                # Try to infer frequency from last few observations
                freq = pd.infer_freq(self.train_data.index[-5:])
                
            if freq is not None:
                forecast_dates = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[1:]
                forecast = pd.Series(forecast, index=forecast_dates)
            else:
                forecast = pd.Series(forecast)
        else:
            forecast = pd.Series(forecast)
        
        return {
            'forecast': forecast,
            'steps': steps
        }
    
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
            forecast_result = self.predict(steps=len(test_data))
            
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