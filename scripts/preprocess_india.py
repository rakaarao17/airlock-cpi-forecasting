import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import os
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

print("Starting India preprocessing...")
np.random.seed(RANDOM_STATE)

market_file = os.path.join(RAW_DATA_PATH, 'market_data_daily.csv')
if not os.path.exists(market_file):
    print("Downloading market data...")
    tickers = {'Crude_Oil_Brent': 'BZ=F', 'Gold': 'GC=F', 'USD_INR': 'INR=X'}
    market_data_daily = yf.download(list(tickers.values()), start='2012-01-01', end='2024-12-31', progress=False)['Close']
    market_data_daily.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
    market_data_daily = market_data_daily.ffill()
    market_data_daily.to_csv(market_file)
    print("Market data saved")

macro_file = os.path.join(RAW_DATA_PATH, 'macro_data_monthly.csv')
if not os.path.exists(macro_file):
    print("Processing macro data...")
    component_files = ['Clothing and footwear.csv', 'Fuel and light.csv', 'Housing.csv', 'Miscellaneous.csv', 'Pan; tobacco; and intoxicants.csv']
    macro_components = []
    for filename in component_files:
        df = pd.read_csv(os.path.join(RAW_DATA_PATH, filename))
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
        df.set_index('Date', inplace=True)
        component_name = filename.replace('.csv', '')
        series = df[df['Description'] == component_name]['Combined'].rename(component_name)
        if not series.empty:
            macro_components.append(series)
    if macro_components:
        macro_data_monthly = pd.concat(macro_components, axis=1)
        macro_data_monthly.to_csv(macro_file)
        print("Macro data saved")

print("Loading targets...")
df_headline = pd.read_csv(os.path.join(RAW_DATA_PATH, 'General Index.csv'))
df_headline['Date'] = pd.to_datetime(df_headline['Year'].astype(str) + '-' + df_headline['Month'], format='%Y-%B')
df_headline.set_index('Date', inplace=True)
Y_headline = df_headline[df_headline['Description'] == 'General Index (All Groups)']['Combined'].rename('Y_headline')

df_food = pd.read_csv(os.path.join(RAW_DATA_PATH, 'Food and beverages.csv'))
df_food['Date'] = pd.to_datetime(df_food['Year'].astype(str) + '-' + df_food['Month'], format='%Y-%B')
df_food.set_index('Date', inplace=True)
Y_food = df_food[df_food['Description'] == 'Food and beverages']['Combined'].rename('Y_food')

targets_df = pd.concat([Y_headline, Y_food], axis=1)
targets_df.index = targets_df.index.to_period('M').to_timestamp('M')
y_final = targets_df.diff(1).dropna()
y_final.columns = ['y_headline_diff', 'y_food_diff']

print("Processing features...")
features_list = []

market_daily = pd.read_csv(market_file, parse_dates=['Date'], index_col='Date')
market_monthly_mean = market_daily.resample('ME').mean()
market_monthly_std = market_daily.resample('ME').std()
market_monthly_mean.columns = [f"{col}_mean" for col in market_monthly_mean.columns]
market_monthly_std.columns = [f"{col}_std" for col in market_monthly_std.columns]
market_monthly = market_monthly_mean.join(market_monthly_std)
features_list.append(market_monthly)

macro_monthly = pd.read_csv(macro_file, parse_dates=['Date'], index_col='Date')
macro_monthly.index = macro_monthly.index.to_period('M').to_timestamp('M')
features_list.append(macro_monthly)

X_raw = pd.concat(features_list, axis=1)
X_raw.index = pd.to_datetime(X_raw.index)
X_lagged = X_raw.shift(PUBLICATION_LAG)

print("Enforcing stationarity...")
X_stationary = pd.DataFrame(index=X_lagged.index)
for col in X_lagged.columns:
    stationary_series, final_name = enforce_stationarity(X_lagged[col], col)
    if stationary_series is not None:
        X_stationary[final_name] = stationary_series

X_stationary['Demo_Shock'] = 0
demo_date = pd.to_datetime('2016-11-30')
if demo_date in X_stationary.index:
    X_stationary.loc[demo_date, 'Demo_Shock'] = 1

X_stationary['GST_Shock'] = 0
gst_date = pd.to_datetime('2017-07-31')
if gst_date in X_stationary.index:
    X_stationary.loc[gst_date, 'GST_Shock'] = 1

print("Saving final data...")
final_dataset = X_stationary.join(y_final)
final_dataset_clean = final_dataset.dropna()

y_final_clean = final_dataset_clean[['y_headline_diff', 'y_food_diff']]
X_final_clean = final_dataset_clean.drop(columns=['y_headline_diff', 'y_food_diff'])

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
X_final_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_final_model_data.csv'))
y_final_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_final_model_data.csv'))

print(f"Complete. X shape: {X_final_clean.shape}, y shape: {y_final_clean.shape}")
print(f"Files saved to {PROCESSED_DATA_PATH}")