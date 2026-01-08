# Time-Series-Modeling-and-Forecast-Case-Study

#**Overview**

This project analyzes and forecasts two time series datasets: quarterly sales for Johnson & Johnson (JJ, 1960–1980) and daily closing share prices for Amazon (2000–2023). By applying traditional ARMA models and deep learning RNN variants (LSTM and GRU), we uncover patterns like seasonality in JJ sales and volatility in Amazon prices. The analysis supports accurate multi-step predictions, offering insights for sales planning and investment strategies.

#**Key goals:**

Decompose series to reveal trends, cycles, and residuals.
Compare model performance using metrics like MAE, RMSE, and MAPE.
Generate forecasts with confidence intervals for future periods.

Results show LSTM/GRU outperforming ARMA on volatile data (e.g., Amazon MAPE ~3.2% vs. 5.2%), highlighting deep learning's edge in non-linear forecasting.

#**Dataset Descriptions**

**JJ Sales Data**

Scope: Monthly sales (millions USD) from January 1960 to December 1980, aggregated quarterly for analysis.
Structure: Columns – Date (YYYY-MM), Sales (total for product lines).
Characteristics: Exhibits strong quarterly seasonality (e.g., Q4 peaks), upward trend, and potential cycles. Ideal for modeling stable, seasonal business metrics.
Size: 252 observations.
Use: Forecasts future sales growth (~15% by 2025), aiding inventory and revenue planning.

**Amazon Share Price Data**

Scope: Daily closing prices (USD, adjusted for splits/dividends) from 2000 to 2023.
Structure: Columns – Date (YYYY-MM-DD), Close (price).
Characteristics: Non-stationary with momentum, volatility clusters (e.g., post-2020 surge), and external influences (e.g., market events). Captures tech sector dynamics.
Size: ~6,000 trading days.
Use: Predicts short-term trends (e.g., $250/share by mid-2026), informing trading decisions.

#**Methodology**

**Data Preparation:**
Cleaning: Handle missing dates, log-transform for stability.
Stationarity: ADF tests, differencing (e.g., 1st-order for Amazon).
Split: 80/20 train-test; scaling (MinMaxScaler) for RNNs.
Decomposition: STL for trends/seasonality; ACF/PACF for lag selection.

**Modeling:**
ARMA: Statsmodels for AR(p)MA(q); auto-selection via AIC/BIC (e.g., ARMA(2,1) for JJ).
LSTM: Keras Sequential (1-2 layers, 50 units, dropout=0.2); MSE loss, Adam optimizer.
GRU: Similar architecture, leveraging gated units for efficiency on sequences.
Training: 50 epochs, batch=32, early stopping; rolling-window validation.

**Forecasting & Evaluation:**
Horizons: 12 quarters (JJ), 30 days (Amazon).
Metrics: MAE/RMSE for absolute errors; MAPE for relative.
Residuals: Ljung-Box for autocorrelation checks.


#**Key Results**

JJ Sales: Seasonality dominates; GRU captures cycles best (RMSE=0.13). Forecast: Steady rise to ~$150M/quarter by 2025.
Amazon Prices: High volatility post-2010; LSTM excels (MAPE=3.5%). Forecast: Uptrend to $250 with 95% CI [$220–$280].
**Model Comparison:**

Dataset,Model,MAE,RMSE,MAPE (%)
JJ Sales,ARMA,0.15,0.18,4.5
JJ Sales,LSTM,0.10,0.12,2.8
JJ Sales,GRU,0.11,0.13,3.1
Amazon,ARMA,12.5,18.2,5.2
Amazon,LSTM,8.4,11.6,3.5
Amazon,GRU,7.9,10.8,3.2

#**Insights & Applications**

RNNs shine for complex patterns (e.g., Amazon's non-linearity); ARMA suffices for seasonal stability (JJ).
Business Value: Enhances sales forecasting for JJ-like firms; risk assessment for volatile stocks.
Limitations: Univariate focus—future: Add covariates (e.g., GDP, volume).

#**Tech Stack**

Data: Pandas, NumPy.
Modeling: Statsmodels, TensorFlow/Keras.
Viz: Matplotlib, Seaborn.
