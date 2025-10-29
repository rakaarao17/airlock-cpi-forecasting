# /code/preprocess_india.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import os
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

print("--- [STARTING INDIA PREPROCESSING] ---")

# --- 1. DEFINE FILE PATHS ---
# Assumes raw data is in a subfolder
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'results' # Save final files to main data folder

# --- 2. ACQUIRE MARKET DATA (if not present) ---
market_file = os.path.join(RAW_DATA_PATH, 'market_data_daily.csv')
if not os.path.exists(market_file):
    print("Market data not found, downloading...")
    try:
        tickers = {'Crude_Oil_Brent': 'BZ=F', 'Gold': 'GC=F', 'USD_INR': 'INR=X'}
        market_data_daily = yf.download(list(tickers.values()), start='2012-01-01', end='2024-12-31')['Close']
        market_data_daily.rename(columns={'BZ=F': 'Crude_Oil_Brent', 'GC=F': 'Gold', 'INR=X': 'USD_INR'}, inplace=True)
        market_data_daily = market_data_daily.ffill()
        market_data_daily.to_csv(market_file)
        print("Market data downloaded and saved.")
    except Exception as e:
        print(f"Error downloading market data: {e}")
        exit()
else:
    print("Market data file found.")

# --- 3. ACQUIRE MACRO DATA (if not present) ---
macro_file = os.path.join(RAW_DATA_PATH, 'macro_data_monthly.csv')
if not os.path.exists(macro_file):
    print("Monthly macro data not found, processing...")
    filenames = [
        'Clothing and footwear.csv', 
        'Fuel and light.csv',
        'Housing.csv',
        'Miscellaneous.csv',
        'Pan; tobacco; and intoxicants.csv'
    ]
    macro_features_list = []
    for f in filenames:
        try:
            f_path = os.path.join(RAW_DATA_PATH, f)
            df = pd.read_csv(f_path)
            df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
            df.set_index('Date', inplace=True)
            
            series_name = f.replace('.csv', '')
            series = df[df['Description'] == series_name]['Combined'].rename(series_name)
            
            if not series.empty:
                macro_features_list.append(series)
            else:
                 # Try general index name
                 desc = "General Index (All Groups)" 
                 series = df[df['Description'] == desc]['Combined'].rename(series_name)
                 if not series.empty:
                     macro_features_list.append(series)
                 else:
                     print(f"Warning: Could not find matching description for {f}")

        except Exception as e:
            print(f"Error processing file {f}: {e}")

    if macro_features_list:
        macro_data_monthly = pd.concat(macro_features_list, axis=1)
        macro_data_monthly = macro_data_monthly.dropna(how='all') 
        macro_data_monthly.to_csv(macro_file)
        print("Monthly macro features processed and saved.")
    else:
        print("ERROR: No macro feature files were loaded.")
        exit()
else:
    print("Monthly macro data file found.")


# --- 4. PROCESS INDIA TARGETS (Y) ---
print("\n--- 4. Processing India Targets (Y) ---")
try:
    df_headline = pd.read_csv(os.path.join(RAW_DATA_PATH, 'General Index.csv'))
    df_headline['Date'] = pd.to_datetime(df_headline['Year'].astype(str) + '-' + df_headline['Month'], format='%Y-%B')
    df_headline.set_index('Date', inplace=True)
    Y_headline = df_headline[df_headline['Description'] == 'General Index (All Groups)']['Combined'].rename('Y_headline')

    df_food = pd.read_csv(os.path.join(RAW_DATA_PATH, 'Food and beverages.csv'))
    df_food['Date'] = pd.to_datetime(df_food['Year'].astype(str) + '-' + df_food['Month'], format='%Y-%B')
    df_food.set_index('Date', inplace=True)
    Y_food = df_food[df_food['Description'] == 'Food and beverages']['Combined'].rename('Y_food')

    targets_df = pd.concat([Y_headline, Y_food], axis=1)
    targets_df.index = targets_df.index.to_period('M').to_timestamp('M') # Standardize to End-of-Month
    
    y_final = targets_df.diff(1).dropna() # Diff of YoY
    y_final.columns = ['y_headline_diff', 'y_food_diff']
    print("India Targets loaded and differenced successfully.")

except Exception as e:
    print(f"Error loading India target files: {e}.")
    exit()

# --- 5. PROCESS INDIA FEATURES (X) ---
print("\n--- 5. Processing India Features (X) ---")
all_features_list = []
try:
    market_daily = pd.read_csv(market_file, parse_dates=['Date'], index_col='Date')
    market_monthly_mean = market_daily.resample('ME').mean()
    market_monthly_std = market_daily.resample('ME').std()
    market_monthly_mean.columns = [f"{col}_mean" for col in market_monthly_mean.columns]
    market_monthly_std.columns = [f"{col}_std" for col in market_monthly_std.columns]
    market_monthly = market_monthly_mean.join(market_monthly_std)
    all_features_list.append(market_monthly)
    print("India Market features aggregated.")
except Exception as e:
    print(f"Could not process 'market_data_daily.csv': {e}")

try:
    macro_monthly = pd.read_csv(macro_file, parse_dates=['Date'], index_col='Date')
    macro_monthly.index = macro_monthly.index.to_period('M').to_timestamp('M')
    all_features_list.append(macro_monthly)
    print("India MoSPI macro features loaded.")
except Exception as e:
    print(f"Could not load 'macro_data_monthly.csv': {e}")

X_raw = pd.concat(all_features_list, axis=1)
X_raw.index = pd.to_datetime(X_raw.index)
X_lagged = X_raw.shift(1) # The "Airlock" lag

print("All India features lagged by 1 month.")

X_stationary = pd.DataFrame(index=X_lagged.index)
discarded_features = []

for col in X_lagged.columns:
    is_stationary, series = run_adf_test(X_lagged[col], col)
    if is_stationary:
        X_stationary[col] = series
    else:
        fixed_series = series.diff(1)
        is_stationary_after_fix, _ = run_adf_test(fixed_series, f"{col}_diff(1)")
        if is_stationary_after_fix:
            X_stationary[f"{col}_diff(1)"] = fixed_series
        else:
            print(f"  ❌❌ {col}: Still non-stationary. DISCARDING.")
            discarded_features.append(col)

print(f"Discarded features: {discarded_features}")

# Add Shocks
demo_date = pd.to_datetime('2016-11-30')
X_stationary['Demo_Shock'] = 0
if demo_date in X_stationary.index: X_stationary.loc[demo_date, 'Demo_Shock'] = 1
gst_date = pd.to_datetime('2017-07-31')
X_stationary['GST_Shock'] = 0
if gst_date in X_stationary.index: X_stationary.loc[gst_date, 'GST_Shock'] = 1
print("Shock variables added.")

# --- 6. FINAL ALIGN & SAVE (India) ---
print("\n--- 6. Final Alignment & Saving (India) ---")
final_dataset = X_stationary.join(y_final)
final_dataset_clean = final_dataset.dropna()

y_final_clean = final_dataset_clean[['y_headline_diff', 'y_food_diff']]
X_final_clean = final_dataset_clean.drop(columns=['y_headline_diff', 'y_food_diff'])

# Save to main data folder
X_final_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_final_model_data.csv'))
y_final_clean.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_final_model_data.csv'))

print(f"Final India X shape: {X_final_clean.shape}, y shape: {y_final_clean.shape}")
print("✅✅ SUCCESS: India data processing complete.")
print(f"Files saved to {PROCESSED_DATA_PATH}")