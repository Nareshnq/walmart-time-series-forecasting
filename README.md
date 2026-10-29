# Walmart Sales Forecasting

A forecasting-methods showcase: weekly Walmart sales are aggregated across all stores and forecast with four univariate models, compared on the same 13-week holdout period.

> Note on the data: this dataset covers February 2010 – October 2012, so the value here is in the forecasting methods and comparison — not in predictions about current sales.

## Objective

Forecast aggregated weekly Walmart sales and compare forecasting methods head-to-head: a simple moving-average baseline, ARIMA, SARIMA (SARIMAX), and Prophet. The notebook also examines seasonality, holiday effects, and how much of the variation in sales external factors can explain — before deciding what the forecasting models should (and should not) use as inputs.

## Dataset

Kaggle Walmart sales dataset: **45 stores, weekly data from February 2010 to October 2012, 6,434 records**.

Columns in `data/Walmart.csv`:

| Column | Description |
| --- | --- |
| `Store` | Store number (1–45) |
| `Date` | Week of sales |
| `Weekly_Sales` | Weekly sales for that store |
| `Holiday_Flag` | Whether the week is a holiday week |
| `Temperature` | Average temperature for the region that week |
| `Fuel_Price` | Fuel price that week |
| `CPI` | Consumer Price Index |
| `Unemployment` | Unemployment rate |

## Methods

1. **Exploratory data analysis (EDA)** — sales by store, distributions, monthly and seasonal patterns, and holiday vs. non-holiday weeks.
2. **External-factor analysis** — correlations and variance explained by temperature, fuel price, CPI, unemployment, and the holiday flag. Together, these factors explain only about **2.5%** of sales variance, so the forecasting models are **univariate**: no exogenous regressors were passed to SARIMAX, and no regressors were added to Prophet.
3. **Decomposition and stationarity** — seasonal decomposition of the aggregated weekly series, plus ADF and ACF/PACF assessment.
4. **Model comparison on the same 13-week holdout** — a 4-week moving-average baseline, ARIMA, SARIMA (with grid search over orders by AIC/BIC), and Prophet, all evaluated on the final 13 weeks of the aggregated series.

All modelling uses data aggregated across all 45 stores into a single weekly series (143 weeks).

## Results

Evaluated on the 13-week holdout (MAPE = mean absolute percentage error):

| Model | MAPE |
| --- | --- |
| Moving Average baseline (MA(4)) | 2.84% |
| ARIMA(5,1,0) | 3.18% |
| SARIMA (3,1,3)x(0,1,2,13) | 9.69% |
| **Prophet** | **2.16%** |

**Prophet is the best model: 2.16% MAPE, 24.1% better than the 2.84% moving-average baseline.** Prophet captures yearly and weekly seasonality most smoothly; SARIMA struggles most with the extreme holiday spikes in the test period.

## How to run

```bash
pip install -r requirements.txt
```

Then open `walmart-time-series-forecasting.ipynb` (from the repo root) and run it top to bottom. The notebook reads the data from the relative path `data/Walmart.csv` — no Google Drive mount or local file path is needed.

## Future work

- **Store-level forecasting** — the current models use data aggregated across all stores; forecasting per store (or hierarchically, with store-level models reconciled to the total) is the natural next step.
- Custom holiday effects in Prophet, and advanced models (XGBoost, LSTM, TFT).
- A production-ready forecasting pipeline around the best model.
