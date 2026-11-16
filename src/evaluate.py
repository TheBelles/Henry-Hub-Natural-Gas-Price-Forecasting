import numpy as np
import pandas as pd

from . import config


def actual_price_ahead(data, horizon, known_until=None):
    actual = data["target"].shift(-horizon)
    if known_until is not None:
        # hide outcomes after known_until (used for the practice test, so it does not peek into the test period)
        outcome_date = data.index + pd.offsets.MonthEnd(horizon)
        actual = actual.where(outcome_date <= known_until)
    return actual


def mape(actual, forecast):
    both = actual.notna() & forecast.notna()
    return float((forecast[both] / actual[both] - 1).abs().mean() * 100)


def accuracy_table(data, forecasts_by_model, dates, known_until=None):
    rows = {}
    for name, forecasts in forecasts_by_model.items():
        scores = {}
        for h in config.HORIZONS:
            actual = actual_price_ahead(data, h, known_until).loc[dates]
            scores[f"{h}m ahead"] = mape(actual, forecasts.loc[dates, f"pred_{h}"])
        scores["average"] = float(np.mean(list(scores.values())))
        rows[name] = scores
    return pd.DataFrame(rows).T


def direction_hit_rate(data, forecasts, dates):
    # % of forecasts that got up/down right compared to today's price
    rates = {}
    for h in config.HORIZONS:
        today = data.loc[dates, "target"]
        actual = actual_price_ahead(data, h).loc[dates]
        forecast = forecasts.loc[dates, f"pred_{h}"]
        known = actual.notna()
        right = np.sign(forecast[known] - today[known]) == np.sign(actual[known] - today[known])
        rates[f"{h}m ahead"] = float(right.mean() * 100)
    return pd.Series(rates)


def turning_points(price, window=2):
    # peak = highest price within +-2 months, trough = lowest
    rows = []
    for i in range(window, len(price) - window):
        nearby = price.iloc[i - window:i + window + 1]
        if price.iloc[i] == nearby.max():
            rows.append({"date": price.index[i], "type": "peak"})
        elif price.iloc[i] == nearby.min():
            rows.append({"date": price.index[i], "type": "trough"})
    return pd.DataFrame(rows).set_index("date")


def turning_point_calls(data, forecasts, dates, window=2):
    # at a peak the 1-month forecast should go down, at a trough it should go up
    points = turning_points(data["target"].dropna(), window)
    points = points[points.index.isin(dates)].copy()
    price_now = data.loc[points.index, "target"]
    forecast = forecasts.loc[points.index, "pred_1"]
    points["price"] = price_now
    points["forecast_next_month"] = forecast
    points["actual_next_month"] = data["target"].shift(-1).loc[points.index]
    points["called"] = np.where(points["type"] == "peak", forecast < price_now, forecast > price_now)
    return points


def reversal_scores(data, forecasts, dates):
    # months where the price changes direction next month
    price = data["target"]
    last_move = np.sign(price - price.shift(1))
    next_move = np.sign(price.shift(-1) - price)
    forecast_move = np.sign(forecasts["pred_1"] - price)

    known = [d for d in dates if pd.notna(price.shift(-1).loc[d])]
    real = (next_move != last_move).loc[known]
    called = (forecast_move != last_move).loc[known]
    hits = int((real & called).sum())
    return {"months": len(known), "real_reversals": int(real.sum()),
            "called_reversals": int(called.sum()), "correct_calls": hits}
