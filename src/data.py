import pandas as pd

from . import config

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FUTURES = ["fut_c1", "fut_c2", "fut_c3", "fut_c4"]


def load_provided():
    train = pd.read_csv(config.RAW_DIR / "data_train.csv", parse_dates=["date"])
    test = pd.read_csv(config.RAW_DIR / "data_test.csv", parse_dates=["date"])
    data = pd.concat([train, test]).set_index("date").sort_index().copy()  # copy() to avoid a pandas warning
    data["is_test"] = data.index > config.TRAIN_END
    return data


def read_eia_table(filename):
    # EIA tables have one row per year and one column per month, so turn them into one long series
    wide = pd.read_csv(config.EXTERNAL_DIR / filename)
    long = wide.melt(id_vars="Year", value_vars=MONTHS, var_name="month", value_name="value")
    long["date"] = pd.to_datetime(long["Year"].astype(str) + "-" + long["month"], format="%Y-%b") + pd.offsets.MonthEnd(0)
    return long.set_index("date")["value"].sort_index()


def load_external():
    external = pd.DataFrame({
        "fut_c1": read_eia_table("eia_futures_contract1.csv"),   # nearest futures contract
        "fut_c2": read_eia_table("eia_futures_contract2.csv"),
        "fut_c3": read_eia_table("eia_futures_contract3.csv"),
        "fut_c4": read_eia_table("eia_futures_contract4.csv"),
        "hh_spot": read_eia_table("eia_henry_hub_spot.csv"),
        "lng_exports": read_eia_table("eia_lng_exports.csv"),     # million cubic feet per month
    })

    europe = pd.read_csv(config.EXTERNAL_DIR / "imf_europe_gas_price.csv", parse_dates=["date"])
    europe["date"] = europe["date"] + pd.offsets.MonthEnd(0)
    external["europe_gas"] = europe.set_index("date")["value"]

    # 2 cells are missing on the EIA site (contract 3 in Jun 2013, contract 4 in Jun 2014),
    # fill them from the other contracts of the same month
    external[FUTURES] = external[FUTURES].interpolate(axis=1)
    return external


def load_all():
    data = load_provided().join(load_external(), how="left")
    data["target"] = data[config.TARGET]

    # before 2013 the target is the spot price, but the test period is a futures price.
    # so I use the futures price for the old years to have one consistent price to learn from
    data["price"] = data["target"].where(data.index >= config.TARGET_DEFINITION_CHANGE, data["fut_c1"])
    return data
