# makes outputs/submission.csv
# run from the project folder:  python -m src.run_pipeline

from . import config
from .data import load_all
from .evaluate import accuracy_table
from .features import build_features
from .model import forecast_all_horizons, futures_curve_forecast, no_change_forecast


def save_submission(forecasts, test_dates):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    submission = forecasts.loc[test_dates, ["pred_1", "pred_2", "pred_3"]]
    submission.to_csv(config.OUTPUT_DIR / "submission.csv", index_label="date",
                      date_format="%Y-%m-%d", float_format="%.6f")
    return submission


def main():
    data = load_all()
    test_dates = data.index[data["is_test"]]

    # train only on the training file (up to Aug 2020), then forecast the 48 test months
    forecasts, _ = forecast_all_horizons(data, build_features(data), config.TRAIN_END)
    save_submission(forecasts, test_dates)

    results = accuracy_table(data, {"No change": no_change_forecast(data),
                                    "Futures curve only": futures_curve_forecast(data),
                                    "Our model": forecasts}, test_dates)
    print("Test MAPE (%):")
    print(results.round(2))
    print("saved", config.OUTPUT_DIR / "submission.csv")


if __name__ == "__main__":
    main()
