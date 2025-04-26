import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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

class LSTMModel:
    """
    LSTM Model for time series forecasting
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        units: List[int] = [50, 50],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        validation_split: float = 0.2
    ):
        """
        Initialize LSTM model
        
        Args:
            sequence_length (int): Number of time steps to use as input features
            units (List[int]): List of LSTM units per layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs for training
            patience (int): Patience for early stopping
            validation_split (float): Fraction of data to use for validation
        """
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
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
        
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture
        
        Args:
            input_shape (Tuple[int, int]): Input shape (sequence_length, n_features)
            
        Returns:
            Sequential: Keras Sequential model
        """
        model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(self.units):
            return_sequences = i < len(self.units) - 1  # Return sequences for all but last layer
            
            if i == 0:
                # First layer needs input shape
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    input_shape=input_shape
                ))
            else:
                # Subsequent layers
                model.add(LSTM(
                    units=units,
                    return_sequences=return_sequences
                ))
            
            # Add batch normalization and dropout for regularization
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Add output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        
        return model
    
    def _create_sequences(
        self, 
        data: np.ndarray, 
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences for LSTM model
        
        Args:
            data (np.ndarray): Time series data
            sequence_length (int): Length of input sequences
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (input sequences) and y (target values)
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def fit(
        self, 
        series: pd.Series, 
        validation_data: Optional[pd.Series] = None,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Fit LSTM model to time series data
        
        Args:
            series (pd.Series): Time series data
            validation_data (Optional[pd.Series]): Validation data (if None, uses validation_split)
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
            X, y = self._create_sequences(scaled_data, self.sequence_length)
            
            # Reshape X for LSTM input [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Prepare validation data if provided
            if validation_data is not None:
                val_data = validation_data.values.reshape(-1, 1)
                scaled_val_data = self.scaler.transform(val_data)
                X_val, y_val = self._create_sequences(scaled_val_data, self.sequence_length)
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
                validation = (X_val, y_val)
                internal_validation_split = 0  # Don't use internal validation split
            else:
                validation = None
                internal_validation_split = self.validation_split
            
            # Build model
            input_shape = (self.sequence_length, 1)  # [time steps, features]
            self.model = self._build_model(input_shape)
            
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
            logger.info(f"Training LSTM model with {len(X)} sequences")
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
            train_pred = self.scaler.inverse_transform(train_pred)
            train_y = self.scaler.inverse_transform(y)
            
            train_rmse = np.sqrt(mean_squared_error(train_y, train_pred))
            train_mae = mean_absolute_error(train_y, train_pred)
            
            # Return fit results
            return {
                'fitted': True,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'epochs_trained': len(self.history.history['loss']),
                'final_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history['val_loss'][-1] if 'val_loss' in self.history.history else None,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'history': self.history.history
            }
            
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {str(e)}")
            self.fitted = False
            return {
                'fitted': False,
                'error': str(e)
            }
    
    def predict(
        self, 
        steps: int = 1,
        use_last_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate forecasts from fitted model
        
        Args:
            steps (int): Number of steps to forecast
            use_last_n (Optional[int]): Use last n observations (default: self.sequence_length)
            
        Returns:
            Dict[str, Any]: Dictionary with forecast results
        """
        if not self.fitted:
            logger.error("Model not fitted. Call fit() first.")
            return {'error': "Model not fitted. Call fit() first."}
        
        try:
            # Determine how many last observations to use
            if use_last_n is None:
                use_last_n = self.sequence_length
            
            # Get the last n observations
            last_data = self.train_data[-use_last_n:].values.reshape(-1, 1)
            
            # Scale the data
            scaled_data = self.scaler.transform(last_data)
            
            # Generate forecasts step by step
            forecasts = []
            
            # Use the latest data as the initial input sequence
            curr_seq = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            for i in range(steps):
                # Predict next step
                next_step = self.model.predict(curr_seq, verbose=0)[0, 0]
                
                # Add prediction to forecasts
                forecasts.append(next_step)
                
                # Update sequence for next prediction
                curr_seq = np.roll(curr_seq, -1, axis=1)
                curr_seq[0, -1, 0] = next_step
            
            # Convert predictions back to original scale
            forecasts = np.array(forecasts).reshape(-1, 1)
            forecasts = self.scaler.inverse_transform(forecasts).flatten()
            
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
                    forecasts = pd.Series(forecasts, index=forecast_dates)
                else:
                    forecasts = pd.Series(forecasts)
            else:
                forecasts = pd.Series(forecasts)
            
            # Return forecast results
            return {
                'forecast': forecasts,
                'steps': steps
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
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
            # Scale data
            data = self.train_data.values.reshape(-1, 1)
            scaled_data = self.scaler.transform(data)
            
            # Create sequences
            X, _ = self._create_sequences(scaled_data, self.sequence_length)
            
            # Reshape X for LSTM input [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Generate predictions
            scaled_preds = self.model.predict(X, verbose=0)
            preds = self.scaler.inverse_transform(scaled_preds).flatten()
            
            # Create Series with appropriate index
            index = self.train_data.index[self.sequence_length:]
            predictions = pd.Series(preds, index=index)
            
            return predictions
            
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
            # Combine train and test data for prediction
            full_data = pd.concat([self.train_data, test_data])
            
            # Scale data
            data = full_data.values.reshape(-1, 1)
            scaled_data = self.scaler.transform(data)
            
            # Create sequences for test period
            X_list, y_list = [], []
            
            # Get sequences that end in test period
            for i in range(len(self.train_data) - self.sequence_length, len(full_data) - self.sequence_length):
                start_idx = i
                end_idx = i + self.sequence_length
                X_list.append(scaled_data[start_idx:end_idx])
                y_list.append(scaled_data[end_idx])
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            # Reshape X for LSTM input [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Generate predictions
            scaled_preds = self.model.predict(X, verbose=0)
            
            # Convert predictions and actual values back to original scale
            preds = self.scaler.inverse_transform(scaled_preds).flatten()
            actuals = self.scaler.inverse_transform(y).flatten()
            
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


def create_lstm_model(
    series: pd.Series,
    validation_data: Optional[pd.Series] = None,
    sequence_length: int = 10,
    layers: List[int] = [64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 10,
    validation_split: float = 0.2,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Create and train LSTM model for time series forecasting
    
    Args:
        series (pd.Series): Time series data
        validation_data (Optional[pd.Series]): Validation data
        sequence_length (int): Number of time steps to use as input features
        layers (List[int]): List of LSTM units per layer
        dropout_rate (float): Dropout rate for regularization
        learning_rate (float): Learning rate for optimizer
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs for training
        patience (int): Patience for early stopping
        validation_split (float): Fraction of data to use for validation
        verbose (int): Verbosity level for training
        
    Returns:
        Dict[str, Any]: Dictionary with model and results
    """
    try:
        # Initialize LSTM model
        model = LSTMModel(
            sequence_length=sequence_length,
            units=layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            validation_split=validation_split
        )
        
        # Fit model
        fit_result = model.fit(series, validation_data=validation_data, verbose=verbose)
        
        # Return model and results
        return {
            'model': model,
            'fit_result': fit_result
        }
        
    except Exception as e:
        logger.error(f"Error creating LSTM model: {str(e)}")
        return {'error': str(e)}