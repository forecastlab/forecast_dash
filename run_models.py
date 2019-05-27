import json
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import norm
import pickle

with open("data_sources.json") as data_sources_json_file:

    data_sources_list = json.load(data_sources_json_file)

    for data_source_dict in data_sources_list:

        print(data_source_dict["title"])

        # Read local CSV that we downloaded earlier
        series_df = pd.read_csv(f"downloads/{data_source_dict['title']}.csv", parse_dates=["date"])

        # Generate forecast
        forecast_len = 8

        AR = 1
        I = 0
        MA = 0
        model = ARIMA(series_df['value'], order=(AR, I, MA))
        arima_results = model.fit(disp=False)
        forecast, std_dev, _ = arima_results.forecast(steps=forecast_len)

        # Store forecast

        forecast_df = pd.concat(
            [
                pd.Series(forecast, name="FORECAST"),
                pd.Series(std_dev, name="STD_DEV"),
                pd.Series(forecast + norm.ppf(0.5 / 2) * std_dev, name="LB_50"),
                pd.Series(forecast - norm.ppf(0.5 / 2) * std_dev, name="UB_50"),
                pd.Series(forecast + norm.ppf(0.25 / 2) * std_dev, name="LB_75"),
                pd.Series(forecast - norm.ppf(0.25 / 2) * std_dev, name="UB_75"),
                pd.Series(forecast + norm.ppf(0.05 / 2) * std_dev, name="LB_95"),
                pd.Series(forecast - norm.ppf(0.05 / 2) * std_dev, name="UB_95"),
            ],
            axis = 1,
        )

        # There is a bug here, the dates for the series are at start of month
        # but the dates for forecasts are at the end of month
        # seek clarification for the reporting periods
        forecast_df.index = pd.date_range(start=series_df['date'].values[-1], periods=forecast_len + 1, freq=data_source_dict["frequency"])[1:]

        data = {
            "series_df": series_df,
            "forecast_df": forecast_df
        }

        f = open(f"forecasts/{data_source_dict['title']}.pkl", 'wb')
        pickle.dump(data, f)
        f.close()