# Data Sources

This project relies on publicly available data from three main sources.

## 1. Indian CPI Data (MoSPI)

The raw data for Indian Headline CPI and its components (Food, Fuel, etc.) was sourced from the Ministry of Statistics and Programme Implementation, Government of India.

* **Source:** MoSPI, All India CPI (Rural, Urban, Combined)
* **Link:** [https://www.mospi.gov.in/](https://www.mospi.gov.in/)
* **Files Used:** `General Index.csv`, `Food and beverages.csv`, `Fuel and light.csv`, `Housing.csv`, `Miscellaneous.csv`, `Pan; tobacco; and intoxicants.csv`. (Note: `Clothing and footwear.csv` was discarded during preprocessing).

## 2. US CPI & Macro Data (FRED)

The raw data for US CPI (Headline, Core) and US Macroeconomic indicators (Industrial Production, PPI) was sourced from the Federal Reserve Economic Data (FRED).

* **Source:** FRED, Federal Reserve Bank of St. Louis
* **Link:** [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/)
* **Series IDs Used:**
    * `CPIAUCSL` (US Headline CPI)
    * `CPILFESL` (US Core CPI)
    * `INDPRO` (US Industrial Production)
    * `PPIACO` (US Producer Price Index)

## 3. Market Data (yfinance)

All daily financial and commodity market data for both India and the US was sourced using the `yfinance` Python library.

* **Source:** Yahoo Finance (via `yfinance` library)
* **Tickers Used:**
    * `BZ=F` (Brent Crude Oil - for India)
    * `CL=F` (WTI Crude Oil - for US)
    * `GC=F` (Gold)
    * `INR=X` (USD/INR Exchange Rate)
    * `DX-Y.NYB` (US Dollar Index - DXY)

---
*This directory should contain the four **final, processed, model-ready** datasets created by the preprocessing scripts:*
* `X_final_model_data.csv` (Processed India Features)
* `y_final_model_data.csv` (Processed India Targets)
* `X_us_final_model_data.csv` (Processed US Features)
* `y_us_final_model_data.csv` (Processed US Targets)