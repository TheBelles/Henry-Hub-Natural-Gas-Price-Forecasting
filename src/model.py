import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from . import config
from .features import market_view

# forecast = futures market view * exp(correction)
# the correction is predicted by a ridge regression, one model for each horizon


def make_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def fit_and_forecast(data, features, horizon, train_end, make=make_model):
    view = market_view(data, horizon)

    # target for the model: how far the real price ended up from the market view (log ratio)
    surprise = np.log(data["price"].shift(-horizon) / view)

    # only train on months where the outcome was already known at train_end
    outcome_date = data.index + pd.offsets.MonthEnd(horizon)
    complete = features.notna().all(axis=1)
    train_rows = complete & surprise.notna() & (outcome_date <= train_end)

    model = make().fit(features[train_rows], surprise[train_rows])

    correction = pd.Series(model.predict(features[complete]), index=features.index[complete])
    forecast = (view * np.exp(correction)).reindex(data.index)
    return forecast, model


def forecast_all_horizons(data, features, train_end, make=make_model):
    forecasts = pd.DataFrame(index=data.index)
    models = {}
    for h in config.HORIZONS:
        forecasts[f"pred_{h}"], models[h] = fit_and_forecast(data, features, h, train_end, make)
    return forecasts, models


# baselines
def no_change_forecast(data):
    return pd.DataFrame({f"pred_{h}": data["price"] for h in config.HORIZONS})


def futures_curve_forecast(data):
    return pd.DataFrame({f"pred_{h}": market_view(data, h) for h in config.HORIZONS})


def coefficients(models, feature_names):
    # the pipeline scales the features first, so divide by the scale to get weights in normal units
    rows = {}
    for h, model in models.items():
        scaler, ridge = model[0], model[1]
        weights = ridge.coef_ / scaler.scale_
        row = dict(zip(feature_names, weights))
        row["usual_gap_to_futures"] = ridge.intercept_ - (weights * scaler.mean_).sum()   # intercept in normal units
        rows[f"{h}m ahead"] = row
    return pd.DataFrame(rows).T
