# Henry Hub gas price forecast

INEEJI ML coding test. At the end of each month, forecast the Henry Hub natural gas price 1, 2 and 3 months ahead for the 48 test months (Sep 2020 to Aug 2024).

## Results

MAPE on the test period (lower is better):

| | 1 month | 2 months | 3 months |
|---|---|---|---|
| No change (next month = this month) | 13.6% | 22.9% | 29.2% |
| Futures curve only | 13.2% | 22.3% | 27.7% |
| My model | 9.8% | 17.7% | 23.6% |

The 1-month forecast gets the direction right 70% of the time, and 8 of the 13 turning points in the test period are called.

## Approach

The target turned out to be the NYMEX front-month futures price (from 2013 on), so I start from what the futures market expects for the next months (EIA data) and train a ridge regression with 3 features to correct it. I picked this model by validating on 2013-2020 inside the training data, where it beat both baselines and more complex models (gradient boosting, random forest, more features).

## Folder structure

```
data/raw/          provided files
data/external/     public data I added (sources in data/external/README.md)
src/
  config.py        paths and settings
  data.py          loads everything into one monthly table
  features.py      model features + the features I tested and dropped
  model.py         model and baselines
  evaluate.py      MAPE, direction, turning points
  run_pipeline.py  makes submission.csv
notebooks/
  data_analysis.ipynb
  model_selection.ipynb
  forecast_and_error_analysis.ipynb
outputs/
  submission.csv
```

## How to run

```
pip install -r requirements.txt
python -m src.run_pipeline
```

This writes `outputs/submission.csv` and prints the test accuracy. It runs offline in a few seconds and gives the same result every time. The notebooks should be run in this order: data_analysis, model_selection, forecast_and_error_analysis.

## submission.csv

One row per forecast date. For example the row `2024-07-31` has the forecasts made on 31 July 2024 for August (`pred_1`), September (`pred_2`) and October 2024 (`pred_3`).

## Avoiding look-ahead

- A forecast made at the end of a month only uses data available on that day. Official statistics (storage, production, ...) come out about 2 months late, so I only use them with a 2-month lag.
- The model was chosen on 2013-2020 validation inside the training file. The test period was only used at the end to measure accuracy.
- The model is trained on the training file only and not updated during the test period.
- In the forecast_and_error_analysis notebook I recompute all 48 forecasts with everything after the forecast date removed. Nothing changes.
