import pandas as pd
import numpy as np
import pandas_datareader.data as web
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
RANDOM_STATE = 42
ADF_SIGNIFICANCE_LEVEL = 0.05
PUBLICATION_LAG = 1

def test_stationarity(series):
    result = adfuller(series.dropna(), autolag='AIC')
    return result[1] <= ADF_SIGNIFICANCE_LEVEL

def enforce_stationarity(series, name):
    if test_stationarity(series):
        return series, name
    diff_series = series.diff(1)
    if test_stationarity(diff_series):
        return diff_series, f"{name}_diff(1)"
    return None, None

print("Starting US preprocessing...")
np.random.seed(RANDOM_STATE)

start_date = datetime.datetime(2012, 1, 1)
end_date = datetime.datetime(2024, 12, 31)

cpi_file = os.path.join(RAW_DATA_PATH, 'us_cpi_raw.csv')
macro_file = os.path.join(RAW_DATA_PATH, 'us_macro_raw.csv')

if not (os.path.exists(cpi_file) and os.path.exists(macro_file)):
    print("Downloading FRED data...")
    fred_codes_map = {
        'CPIAUCSL': 'CPI_Headline',
        'CPILFESL': 'CPI_Core',
        'INDPRO': 'IP',
        'PPIACO': 'PPI'
    }
    fred_data = web.DataReader(list(fred_codes_map.keys()), 'fred', start_date, end_date)
    fred_data = fred_data.rename(columns=fred_codes_map)
    us_cpi_raw = fred_data[['CPI_Headline', 'CPI_Core']]
    us_macro_raw = fred_data[['IP', 'PPI']]
    us_cpi_raw.to_csv(cpi_file)
    us_macro_raw.to_csv(macro_file)
    print("FRED data saved")

market_file = os.path.join(RAW_DATA_PATH, 'us_market_data_daily.csv')
if not os.path.exists(market_file):
    print("Downloading market data...")
    yf_tickers = {'Crude_Oil_WTI': 'CL=F', 'Gold': 'GC=F', 'USD_Index': 'DX-Y.NYB'}
    market_data_daily_us = yf.download(list(yf_tickers.values()), start=start_date, end=end_date, progress=False)['Close']
    market_data_daily_us.rename(columns={'CL=F': 'Crude_Oil_WTI', 'GC=F': 'Gold', 'DX-Y.NYB': 'USD_Index'}, inplace=True)
    market_data_daily_us = market_data_daily_us.ffill()
    market_data_daily_us.to_csv(market_file)
    print("Market data saved")

print("Loading targets...")
us_cpi_raw = pd.read_csv(cpi_file, index_col=0, parse_dates=True)
us_cpi_yoy = us_cpi_raw.pct_change(periods=12) * 100
us_cpi_yoy.columns = ['Y_us_headline_YoY', 'Y_us_core_YoY']

y_us_final = pd.DataFrame(index=us_cpi_yoy.index)

if not test_stationarity(us_cpi_yoy['Y_us_headline_YoY']):
    y_us_final['y_us_headline_diff'] = us_cpi_yoy['Y_us_headline_YoY'].diff(1)
else:
    y_us_final['y_us_headline_yoy'] = us_cpi_yoy['Y_us_headline_YoY']

if not test_stationarity(us_cpi_yoy['Y_us_core_YoY']):
    y_us_final['y_us_core_diff'] = us_cpi_yoy['Y_us_core_YoY'].diff(1)
else:
    y_us_final['y_us_core_yoy'] = us_cpi_yoy['Y_us_core_YoY']

y_us_final.index = y_us_final.index.to_period('M').to_timestamp('M')
y_us_final = y_us_final.dropna()

print("Processing features...")
all_us_features_list = []

us_market_daily = pd.read_csv(market_file, index_col=0, parse_dates=True)
us_market_monthly_mean = us_market_daily.resample('ME').mean()
us_market_monthly_std = us_market_daily.resample('ME').std()
us_market_monthly_mean.columns = [f"{col}_mean" for col in us_market_monthly_mean.columns]
us_market_monthly_std.columns = [f"{col}_std" for col in us_market_monthly_std.columns]
us_market_monthly = us_market_monthly_mean.join(us_market_monthly_std)
all_us_features_list.append(us_market_monthly)

us_macro_raw = pd.read_csv(macro_file, index_col=0, parse_dates=True)
us_macro_raw.index = us_macro_raw.index.to_period('M').to_timestamp('M')
all_us_features_list.append(us_macro_raw)

X_us_raw = pd.concat(all_us_features_list, axis=1)
X_us_raw.index = pd.to_datetime(X_us_raw.index)
X_us_lagged = X_us_raw.shift(PUBLICATION_LAG)

print("Enforcing stationarity...")
X_us_stationary = pd.DataFrame(index=X_us_lagged.index)
for col in X_us_lagged.columns:
    stationary_series, final_name = enforce_stationarity(X_us_lagged[col], col)
    if stationary_series is not None:
        X_us_stationary[final_name] = stationary_series

X_us_stationary['Covid_Shock'] = 0
covid_date = pd.to_datetime('2020-03-31')
if covid_date in X_us_stationary.index:
    X_us_stationary.loc[covid_date, 'Covid_Shock'] = 1

print("Saving final data...")
final_us_dataset = X_us_stationary.join(y_us_final)
final_us_dataset_clean = final_us_dataset.dropna()

y_us_final_clean = final_us_dataset_clean[y_us_final.columns]
X_us_final_clean = final_us_dataset_clean.drop(columns=y_us_final.columns)

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
X_us_final_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_us_final_model_data.csv'))
y_us_final_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_us_final_model_data.csv'))

print(f"Complete. X shape: {X_us_final_clean.shape}, y shape: {y_us_final_clean.shape}")
print(f"Files saved to {PROCESSED_DATA_PATH}")