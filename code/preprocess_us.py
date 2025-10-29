# /code/preprocess_us.py
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import os
import datetime
import warnings
warnings.filterwarnings('ignore')

# Helper Function for ADF Test
def run_adf_test(series, name):
    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        if p_value <= 0.05:
            print(f"  ✅ {name}: Stationary (p-value: {p_value:.4f})")
            return True, series
        else:
            print(f"  🚨 {name}: NON-STATIONARY (p-value: {p_value:.4f}) -> Fixing...")
            return False, series
    except Exception as e:
        print(f"  ❌ {name}: Error in ADF test: {e}")
        return False, series

print("--- [STARTING US PREPROCESSING] ---")

# --- 1. DEFINE FILE PATHS ---
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'results'
start_date = datetime.datetime(2012, 1, 1)
end_date = datetime.datetime(2024, 12, 31)

# --- 2. ACQUIRE US CPI & MACRO (FRED) (if not present) ---
cpi_file = os.path.join(RAW_DATA_PATH, 'us_cpi_raw.csv')
macro_file = os.path.join(RAW_DATA_PATH, 'us_macro_raw.csv')

if not (os.path.exists(cpi_file) and os.path.exists(macro_file)):
    print("US FRED data not found, downloading...")
    fred_codes_map = {
        'CPIAUCSL': 'CPI_Headline', 'CPILFESL': 'CPI_Core',
        'INDPRO': 'IP', 'PPIACO': 'PPI'
    }
    try:
        fred_data = web.DataReader(list(fred_codes_map.keys()), 'fred', start_date, end_date)
        fred_data = fred_data.rename(columns=fred_codes_map)
        
        us_cpi_raw = fred_data[['CPI_Headline', 'CPI_Core']]
        us_macro_raw = fred_data[['IP', 'PPI']]
        
        us_cpi_raw.to_csv(cpi_file)
        us_macro_raw.to_csv(macro_file)
        print("US CPI and Macro data downloaded and saved.")
    except Exception as e:
        print(f"ERROR fetching FRED data: {e}")
        exit()
else:
    print("US FRED data files found.")

# --- 3. ACQUIRE US MARKET DATA (if not present) ---
market_file = os.path.join(RAW_DATA_PATH, 'us_market_data_daily.csv')
if not os.path.exists(market_file):
    print("US Market data not found, downloading...")
    yf_tickers = {'Crude_Oil_WTI': 'CL=F', 'Gold': 'GC=F', 'USD_Index': 'DX-Y.NYB'}
    try:
        market_data_daily_us = yf.download(list(yf_tickers.values()), start=start_date, end=end_date)['Close']
        market_data_daily_us.rename(columns={'CL=F': 'Crude_Oil_WTI', 'GC=F': 'Gold', 'DX-Y.NYB': 'USD_Index'}, inplace=True)
        market_data_daily_us = market_data_daily_us.ffill()
        market_data_daily_us.to_csv(market_file)
        print("US Market data downloaded and saved.")
    except Exception as e:
        print(f"ERROR fetching yfinance data: {e}")
        exit()
else:
    print("US Market data file found.")

# --- 4. PROCESS US TARGETS (Y_us) ---
print("\n--- 4. Processing US Targets (Y) ---")
try:
    us_cpi_raw = pd.read_csv(cpi_file, index_col=0, parse_dates=True)
    us_cpi_yoy = us_cpi_raw.pct_change(periods=12) * 100
    us_cpi_yoy.columns = ['Y_us_headline_YoY', 'Y_us_core_YoY']
    
    y_us_final = pd.DataFrame(index=us_cpi_yoy.index)
    
    stationary_headline_yoy, series_h = run_adf_test(us_cpi_yoy['Y_us_headline_YoY'], 'Y_us_headline_YoY')
    if not stationary_headline_yoy:
        y_us_final['y_us_headline_diff'] = series_h.diff(1)
    else: y_us_final['y_us_headline_yoy'] = series_h

    stationary_core_yoy, series_c = run_adf_test(us_cpi_yoy['Y_us_core_YoY'], 'Y_us_core_YoY')
    if not stationary_core_yoy:
        y_us_final['y_us_core_diff'] = series_c.diff(1)
    else: y_us_final['y_us_core_yoy'] = series_c

    y_us_final.index = y_us_final.index.to_period('M').to_timestamp('M')
    y_us_final = y_us_final.dropna()
    print("US Targets loaded and differenced successfully.")

except Exception as e:
    print(f"Error processing US CPI: {e}.")
    exit()

# --- 5. PROCESS US FEATURES (X_us) ---
print("\n--- 5. Processing US Features (X) ---")
all_us_features_list = []
try:
    us_market_daily = pd.read_csv(market_file, index_col=0, parse_dates=True)
    us_market_monthly_mean = us_market_daily.resample('ME').mean()
    us_market_monthly_std = us_market_daily.resample('ME').std()
    us_market_monthly_mean.columns = [f"{col}_mean" for col in us_market_monthly_mean.columns]
    us_market_monthly_std.columns = [f"{col}_std" for col in us_market_monthly_std.columns]
    us_market_monthly = us_market_monthly_mean.join(us_market_monthly_std)
    all_us_features_list.append(us_market_monthly)
    print("US Market features aggregated.")
except Exception as e:
    print(f"Could not process 'us_market_data_daily.csv': {e}")

try:
    us_macro_raw = pd.read_csv(macro_file, index_col=0, parse_dates=True)
    us_macro_raw.index = us_macro_raw.index.to_period('M').to_timestamp('M')
    all_us_features_list.append(us_macro_raw)
    print("US Macro features (IP, PPI) loaded.")
except Exception as e:
    print(f"Could not load 'us_macro_raw.csv': {e}")

X_us_raw = pd.concat(all_us_features_list, axis=1)
X_us_raw.index = pd.to_datetime(X_us_raw.index)
X_us_lagged = X_us_raw.shift(1)
print("All US features lagged by 1 month.")

X_us_stationary = pd.DataFrame(index=X_us_lagged.index)
discarded_us_features = []

for col in X_us_lagged.columns:
    is_stationary, series = run_adf_test(X_us_lagged[col], col)
    if is_stationary:
        X_us_stationary[col] = series
    else:
        fixed_series = series.diff(1)
        is_stationary_after_fix, _ = run_adf_test(fixed_series, f"{col}_diff(1)")
        if is_stationary_after_fix:
            X_us_stationary[f"{col}_diff(1)"] = fixed_series
        else:
            print(f"  ❌❌ {col}: Still non-stationary. DISCARDING.")
            discarded_us_features.append(col)

print(f"Discarded features: {discarded_us_features}")

# Add Shock
covid_date = pd.to_datetime('2020-03-31')
X_us_stationary['Covid_Shock'] = 0
if covid_date in X_us_stationary.index: X_us_stationary.loc[covid_date, 'Covid_Shock'] = 1
print("Shock variable added.")

# --- 6. FINAL ALIGNMENT & SAVE (US) ---
print("\n--- 6. Final Alignment & Saving (US) ---")
final_us_dataset = X_us_stationary.join(y_us_final)
final_us_dataset_clean = final_us_dataset.dropna()

y_us_final_clean = final_us_dataset_clean[y_us_final.columns] # Select only target columns
X_us_final_clean = final_us_dataset_clean.drop(columns=y_us_final.columns)

# Save to main data folder
X_us_final_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_us_final_model_data.csv'))
y_us_final_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_us_final_model_data.csv'))

print(f"Final US X shape: {X_us_final_clean.shape}, y shape: {y_us_final_clean.shape}")
print("✅✅ SUCCESS: US data processing complete.")
print(f"Files saved to {PROCESSED_DATA_PATH}")