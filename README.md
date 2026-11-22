# Bitcoin Price Prediction: What Can and Cannot Be Predicted

Can machine learning predict Bitcoin? I tested three prediction targets against honest baselines, using a time-ordered split (train on 2013 to 2018, test on 2019 to May 2021).

## Summary

| Target | Result |
|---|---|
| **Price level** | Looks accurate (about 2.6% average error) but does **not** beat the naive "tomorrow = today" forecast (2.47%) |
| **Daily return** | No model beats a "predict 0%" baseline; R-squared is at or below zero |
| **Direction (up/down)** | 54.8% to 56.6% accuracy vs 54.4% for always guessing "up"; not statistically significant (p = 0.43 and 0.10) |
| **Trading backtest** | Model after fees turned $1 into $11.55 vs $15.33 for buy and hold, so no evidence of an edge |
| **Volatility (next 7 days)** | **Partly predictable**: ridge regression cut error by about 19% versus a "last week" baseline (R-squared 0.11 vs -0.34) |

**Takeaway:** price direction is essentially unpredictable with price-based features, but volatility forecasts add real value for risk management.

## What I did

- Checked the data (2,787 daily rows, no gaps, no missing values) and examined returns, which are fat-tailed.
- Built features from past returns, 5-day and 20-day momentum, rolling volatility, daily range and day of week, using only information available at each day's close.
- Used a **time-ordered split** (no shuffling) and dropped the last 7 training rows to avoid leakage from the 7-day volatility target.
- Compared ridge regression, logistic regression and random forests against naive baselines, with walk-forward cross-validation and a binomial significance test for direction accuracy.
- Ran a simple backtest with a 0.1% fee per position change.

## Files

| File | Description |
|---|---|
| `bitcoin_price_prediction.ipynb` | Full notebook with code, charts and conclusions |
| `btc_usd_daily.csv` | Daily BTC-USD prices, 1 Oct 2013 to 18 May 2021 (CoinDesk) |

## Run it

```
pip install pandas numpy matplotlib scipy scikit-learn jupyter
jupyter notebook bitcoin_price_prediction.ipynb
```
Keep the CSV in the same folder as the notebook.

## Limitations

The data ends in May 2021 and covers one asset. Only price-based features are used (no volume, news or on-chain data). The backtest is simplified (no slippage or taxes). This is a learning project and not investment advice.

## Tools

Python, pandas, NumPy, SciPy, scikit-learn, matplotlib
