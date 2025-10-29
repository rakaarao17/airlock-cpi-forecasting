# /code/run_validation_gauntlet.py
import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from scipy import stats
import warnings
import traceback
import os

# Try to import the correct dm_test
try:
    from statsmodels.stats.diagnostic import dm_test
    USE_STATSMODELS_DM = True
    print("Using dm_test from statsmodels.stats.diagnostic.")
except ImportError:
    try:
        from statsmodels.sandbox.stats.diagnostic import dm_test
        USE_STATSMODELS_DM = True
        print("Using dm_test from statsmodels.sandbox.stats.diagnostic.")
    except ImportError:
        print("Warning: statsmodels dm_test not found. Using manual implementation (no HAC).")
        USE_STATSMODELS_DM = False
        
warnings.filterwarnings('ignore')

# Manual DM Test fallback
def diebold_mariano_test_manual(e1, e2, h=1, power=2):
    e1, e2 = np.array(e1), np.array(e2)
    d = np.abs(e1)**power - np.abs(e2)**power
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) # Simple variance
    n = len(d)
    if var_d == 0: return 0.0, 1.0
    dm_stat = mean_d / np.sqrt(var_d / n)
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df))
    return dm_stat, p_value

# --- 1. DEFINE PARAMETERS ---
print("--- [STARTING VALIDATION GAUNTLET] ---")
DATA_PATH = 'results'
RESULTS_PATH = 'results'
os.makedirs(RESULTS_PATH, exist_ok=True) # Create results folder if it doesn't exist

# --- List of all experiments to run ---
experiments = [
    # H=1 Runs
    {'country': 'India', 'X_file': 'X_final_model_data.csv', 'y_file': 'y_final_model_data.csv', 'target': 'y_headline_diff', 'H': 1},
    {'country': 'India', 'X_file': 'X_final_model_data.csv', 'y_file': 'y_final_model_data.csv', 'target': 'y_food_diff', 'H': 1},
    {'country': 'US', 'X_file': 'X_us_final_model_data.csv', 'y_file': 'y_us_final_model_data.csv', 'target': 'y_us_headline_diff', 'H': 1},
    {'country': 'US', 'X_file': 'X_us_final_model_data.csv', 'y_file': 'y_us_final_model_data.csv', 'target': 'y_us_core_diff', 'H': 1},
    # H=3 Runs
    {'country': 'India', 'X_file': 'X_final_model_data.csv', 'y_file': 'y_final_model_data.csv', 'target': 'y_headline_diff', 'H': 3},
    {'country': 'India', 'X_file': 'X_final_model_data.csv', 'y_file': 'y_final_model_data.csv', 'target': 'y_food_diff', 'H': 3},
    {'country': 'US', 'X_file': 'X_us_final_model_data.csv', 'y_file': 'y_us_final_model_data.csv', 'target': 'y_us_headline_diff', 'H': 3},
    {'country': 'US', 'X_file': 'X_us_final_model_data.csv', 'y_file': 'y_us_final_model_data.csv', 'target': 'y_us_core_diff', 'H': 3},
]

# Store all results here
all_results = []

# ==========================================================
# --- 2. MAIN EXPERIMENT LOOP ---
# ==========================================================
for exp in experiments:
    COUNTRY_LABEL = exp['country']
    X_FILENAME = os.path.join(DATA_PATH, exp['X_file'])
    Y_FILENAME = os.path.join(DATA_PATH, exp['y_file'])
    TARGET_NAME = exp['target']
    H = exp['H'] # Forecast Horizon

    print(f"\n--- RUNNING EXPERIMENT ---")
    print(f"Country: {COUNTRY_LABEL}, Target: {TARGET_NAME}, Horizon: H={H}")

    try:
        X_all = pd.read_csv(X_FILENAME, index_col=0, parse_dates=True)
        y_all = pd.read_csv(Y_FILENAME, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading data for {COUNTRY_LABEL}: {e}")
        continue

    if TARGET_NAME not in y_all.columns:
        print(f"ERROR: Target '{TARGET_NAME}' not found in {Y_FILENAME}.")
        continue

    # --- 2.1 Prepare Data for Horizon ---
    if H == 1:
        y_target = y_all[TARGET_NAME]
    else:
        # Direct Strategy: Shift target *back* by H-1 steps
        y_target = y_all[TARGET_NAME].shift(-(H-1))
    
    # Align X and y, and drop NaNs created by the shift
    y_target, X = y_target.align(X_all, join='inner')
    y_target = y_target.dropna()
    X = X.loc[y_target.index]
    
    print(f"Aligned X shape: {X.shape}, Aligned y shape: {y_target.shape}")

    # --- 2.2 TimeSeries Split ---
    if COUNTRY_LABEL == "India":
        N_SPLITS = int(len(X) * 0.2); N_SPLITS = min(max(N_SPLITS, 20), 30)
    else: # US
        N_SPLITS = int(len(X) * 0.25); N_SPLITS = min(max(N_SPLITS, 25), 40)

    # Adjust splits for H=3 (fewer data points)
    if H > 1:
        if COUNTRY_LABEL == "India": N_SPLITS = 24
        else: N_SPLITS = 35

    tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=1)
    print(f"Using {N_SPLITS} splits...")

    # --- Containers ---
    predictions = []
    actuals = []
    model_names = ['Naïve', 'SARIMA', 'SARIMAX', 'Lasso', 'RandomForest', 'XGBoost']
    errors = {name: [] for name in model_names}

    # --- 2.3 Hyperparameter Grids ---
    nested_tscv = TimeSeriesSplit(n_splits=5, test_size=1)
    pipe_lasso = Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=42, max_iter=10000))])
    grid_lasso = {'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
    pipe_rf = Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42, n_jobs=-1))])
    grid_rf = {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5]}
    pipe_xgb = Pipeline([('scaler', StandardScaler()), ('model', XGBRegressor(random_state=42, n_jobs=-1))])
    grid_xgb = {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5], 'model__learning_rate': [0.01, 0.1]}

    # ==========================================================
    # --- 3. Walk-Forward Loop ---
    # ==========================================================
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"  --- Fold {i+1}/{N_SPLITS} ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_target.iloc[train_index], y_target.iloc[test_index]
        actual = y_test.iloc[0]
        actuals.append(actual)
        fold_preds = {}

        # --- Model 0: Naïve ---
        pred_naive = y_train.iloc[-1]
        fold_preds['Naïve'] = pred_naive
        errors['Naïve'].append(actual - pred_naive)

        # --- Model 1: SARIMA ---
        try:
            model_sarima = pm.auto_arima(y_train.values, seasonal=True, m=12, suppress_warnings=True, error_action='raise', stepwise=True, n_jobs=-1)
            pred_sarima = model_sarima.predict(n_periods=H)[-1] # Get final forecast
        except Exception as e:
            pred_sarima = 0.0
        fold_preds['SARIMA'] = pred_sarima
        errors['SARIMA'].append(actual - pred_sarima)

        # --- Model 2: SARIMAX ---
        try:
            model_sarimax = pm.auto_arima(y_train.values, exogenous=X_train.values, seasonal=True, m=12, suppress_warnings=True, error_action='raise', stepwise=True, n_jobs=-1)
            # Use autoregressive forecast only, as future X is unknown
            pred_sarimax = model_sarimax.predict(n_periods=H)[-1]
        except Exception as e:
            pred_sarimax = 0.0
        fold_preds['SARIMAX'] = pred_sarimax
        errors['SARIMAX'].append(actual - pred_sarimax)

        # --- Model 3: Lasso (Direct Forecast) ---
        grid_search_lasso = GridSearchCV(pipe_lasso, grid_lasso, cv=nested_tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid_search_lasso.fit(X_train, y_train)
        pred_lasso = grid_search_lasso.predict(X_test)[0]
        fold_preds['Lasso'] = pred_lasso
        errors['Lasso'].append(actual - pred_lasso)

        # --- Model 4: RandomForest (Direct Forecast) ---
        grid_search_rf = GridSearchCV(pipe_rf, grid_rf, cv=nested_tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        pred_rf = grid_search_rf.predict(X_test)[0]
        fold_preds['RandomForest'] = pred_rf
        errors['RandomForest'].append(actual - pred_rf)

        # --- Model 5: XGBoost (Direct Forecast) ---
        grid_search_xgb = GridSearchCV(pipe_xgb, grid_xgb, cv=nested_tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid_search_xgb.fit(X_train, y_train)
        pred_xgb = grid_search_xgb.predict(X_test)[0]
        fold_preds['XGBoost'] = pred_xgb
        errors['XGBoost'].append(actual - pred_xgb)

        predictions.append(fold_preds)

    # ==========================================================
    # --- 4. Evaluation and DM Tests for this Experiment ---
    # ==========================================================
    print(f"\n--- RESULTS FOR: {COUNTRY_LABEL} / {TARGET_NAME} / H={H} ---")
    df_preds = pd.DataFrame(predictions)
    df_actuals = pd.Series(actuals, name="Actual")

    for model_name in model_names:
        rmse = np.sqrt(mean_squared_error(df_actuals, df_preds[model_name]))
        mae = mean_absolute_error(df_actuals, df_preds[model_name])
        
        # Store results
        result_row = {
            'Country': COUNTRY_LABEL,
            'Target': TARGET_NAME,
            'Horizon': H,
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae
        }
        
        # Add DM Test results relative to SARIMA
        if model_name not in ['Naïve', 'SARIMA']:
            # *** FIX START ***
            # Define test_type here so it's always available for the column name
            test_type = "(HAC)" if USE_STATSMODELS_DM else "(No HAC)"
            # *** FIX END ***

            dm_stat, dm_pvalue, dm_verdict = np.nan, np.nan, "N/A"
            if 'SARIMA' in errors and errors['SARIMA']:
                errors_sarima = np.array(errors['SARIMA'])
                errors_model = np.array(errors[model_name])
                loss1, loss2 = errors_sarima**2, errors_model**2
                
                if np.allclose(loss1, loss2):
                    dm_verdict = "Identical Errors"
                else:
                    try:
                        dm_func = dm_test if USE_STATSMODELS_DM else diebold_mariano_test_manual
                        dm_stat, dm_pvalue = dm_func(loss1, loss2, h=H, power=2)
                        dm_verdict = f"p={dm_pvalue:.4f}"
                        if dm_pvalue < 0.05:
                            dm_verdict += " (Sig. Better)" if dm_stat > 0 else " (Sig. Worse)"
                        else:
                            dm_verdict += " (No Sig. Diff.)"
                    except Exception as e:
                        dm_verdict = f"Error: {e}"
            result_row[f'DM_vs_SARIMA_{test_type}'] = dm_verdict
        
        all_results.append(result_row)

# --- 5. Save All Results to CSV ---
print("\n--- [FINAL] SAVING ALL RESULTS TO CSV ---")
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by=['Country', 'Target', 'Horizon', 'RMSE'])
results_file = os.path.join(RESULTS_PATH, 'metrics_summary.csv')
results_df.to_csv(results_file, index=False)

print(results_df)
print(f"\n✅✅ SUCCESS: All experiments complete.")
print(f"Final results saved to: {results_file}")
print("--- SCRIPT COMPLETE ---")