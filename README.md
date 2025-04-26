# Financial Time Series Analysis Dashboard

This dashboard allows you to analyze securities using different time series models (ARIMA, LSTM, D-Linear, N-Linear, TimeGPT) and visualize potential investment outcomes.

## Features

- **Data Retrieval**: Fetch financial data (stocks, forex, crypto) from Alpha Vantage API
- **Technical Analysis**: Calculate and visualize various technical indicators
  - Moving Averages (SMA, EMA)
  - Oscillators (RSI, Stochastic)
  - Volume Indicators (OBV, VWAP)
- **Time Series Analysis**:
  - Decomposition (trend, seasonality, residual)
  - Rolling statistics
  - Statistical tests (ADF, KPSS)
  - ACF and PACF plots
- **Forecasting Models**:
  - ARIMA (with auto parameter selection)
  - LSTM (deep learning)
  - D-Linear (decomposition-based linear model)
  - N-Linear (normalization-based linear model)
  - TimeGPT (optional, requires API key)
- **Model Comparison**:
  - Performance metrics (RMSE, MAE, MAPE, R²)
  - Forecast visualization
  - Investment simulation

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/financial-time-series-dashboard.git
   cd financial-time-series-dashboard
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. For TimeGPT functionality (optional):
   ```
   pip install nixtla
   ```

## Usage

1. Get an Alpha Vantage API key:
   - Sign up at [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
   - Free tier allows up to 5 API requests per minute and 500 requests per day

2. (Optional) Get a TimeGPT API key:
   - Sign up at [Nixtla](https://nixtla.io/)

3. Run the application:
   ```
   streamlit run app.py
   ```

4. In the web interface:
   - Enter your API key(s) in the sidebar
   - Select a security type and enter a ticker symbol
   - Choose a time period and analysis type
   - Click "Analyze" to run the analysis

## Project Structure

```
financial-time-series-dashboard/
├── app.py                # Main Streamlit application
├── data/                 # Data handling components
│   ├── data_loader.py    # Functions to fetch data from Alpha Vantage
│   └── data_processor.py # Data preprocessing functions
├── models/               # Time series forecasting models
│   ├── arima_model.py    # ARIMA implementation
│   ├── lstm_model.py     # LSTM implementation
│   ├── linear_models.py  # D-Linear and N-Linear models
│   └── timegpt_model.py  # TimeGPT integration
├── analysis/             # Analysis components
│   ├── decomposition.py  # Time series decomposition
│   ├── statistical_tests.py # ACF, PACF, ADF tests
│   └── indicators/       # Technical indicators
│       ├── moving_averages.py
│       ├── oscillators.py
│       └── volume.py
└── visualization/        # Visualization components
    ├── price_plots.py
    └── performance_plots.py
```

## API Keys and Usage Limits

- **Alpha Vantage**: Free tier allows 5 API calls per minute and 500 per day
- **TimeGPT**: Refer to [Nixtla documentation](https://docs.nixtla.io/) for current usage limits

## Requirements

- Python 3.8 or higher
- Internet connection for API access

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Alpha Vantage](https://www.alphavantage.co/) for financial data API
- [Nixtla](https://nixtla.io/) for TimeGPT
- [Streamlit](https://streamlit.io/) for the web application framework
- [TensorFlow](https://www.tensorflow.org/) for deep learning models
- [statsmodels](https://www.statsmodels.org/) for statistical models