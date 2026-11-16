import numpy as np
import pandas as pd

from . import config

# every feature for month t can only use data known at the end of month t

FEATURES = ["month_end_move", "price_change_1m", "oil_change_3m"]

# features I tried in model selection but did not keep
CANDIDATES = ["storage_surplus", "storage_surplus_change_3m", "production_growth", "consumption_vs_normal",
              "weather_demand", "rig_count_change_6m", "lng_export_share_change", "europe_price_gap",
              "dollar_change_1m", "price_vs_12m_average", "season"]


def same_month_average(series, years=5):
    # average of the same month in the previous `years` years, e.g. storage vs its 5-year average
    past = [series.shift(12 * k) for k in range(1, years + 1)]
    return pd.concat(past, axis=1).mean(axis=1)


def storage_surplus(data, delay_months=0):
    storage = data["Total_Natural_Gas_Underground_Storage_Volume_USA"].shift(delay_months)
    return (storage - same_month_average(storage)) / 1e6   # trillion cubic feet


def curve_slope(data, horizon):
    # how much more (or less) the futures market pays for gas `horizon` months further out, as a log ratio
    slope = np.log(data[f"fut_c{horizon + 1}"] / data["fut_c1"])
    # no futures data after April 2024, so for those months I take the usual slope of that calendar month
    return slope.fillna(same_month_average(slope))


def market_view(data, horizon):
    # the price the futures market expects in `horizon` months
    return data["price"] * np.exp(curve_slope(data, horizon))


def build_features(data):
    features = pd.DataFrame(index=data.index)

    # BCOMNG_INDX is the gas futures index on the LAST day of the month, while the target is a monthly average.
    # if gas jumps at the end of the month, the average catches up next month
    features["month_end_move"] = np.log(data["BCOMNG_INDX"]).diff()

    # how much the monthly average already moved (part of the jump is already in it)
    features["price_change_1m"] = np.log(data["price"]).diff()

    # oil trend over 3 months
    features["oil_change_3m"] = np.log(data["WTI_Crude_Oil"]).diff(3)

    return features


def build_candidate_features(data):
    lag = config.PUBLICATION_LAG_MONTHS
    c = pd.DataFrame(index=data.index)
    log_price = np.log(data["price"])

    c["storage_surplus"] = storage_surplus(data, delay_months=lag)
    c["storage_surplus_change_3m"] = c["storage_surplus"].diff(3)

    production = data["Total_Natural_Gas_Marketed_Production_USA"].shift(lag)
    c["production_growth"] = np.log(production / production.shift(12))

    consumption = data["Natural_Gas_Total_Consumption_USA"].shift(lag)
    c["consumption_vs_normal"] = np.log(consumption / same_month_average(consumption))

    # temperature vs normal. cold winter or hot summer -> more gas demand. no lag, temperature is known right away
    temperature = (data["high_temperature"] + data["low_temperature"]) / 2
    anomaly = temperature - same_month_average(temperature, years=10)
    month = data.index.month
    summer = np.isin(month, [6, 7, 8, 9]).astype(float)
    winter = np.isin(month, [11, 12, 1, 2, 3]).astype(float)
    c["weather_demand"] = anomaly * (summer - winter)

    rigs = data["Natural_Gas_Rotary_Rig_Count_USA"].shift(lag)
    c["rig_count_change_6m"] = np.log(rigs / rigs.shift(6))

    lng_share = (data["lng_exports"] / data["Total_Natural_Gas_Marketed_Production_USA"]).shift(lag)
    c["lng_export_share_change"] = lng_share - lng_share.shift(12)

    c["europe_price_gap"] = np.log(data["europe_gas"] / data["price"])
    c["dollar_change_1m"] = np.log(data["DXY_INDX"]).diff()
    c["price_vs_12m_average"] = log_price - log_price.rolling(12).mean()
    c["season_sin"] = np.sin(2 * np.pi * month / 12)
    c["season_cos"] = np.cos(2 * np.pi * month / 12)
    return c
