import json
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA as smARIMA
from scipy.stats import norm
import pickle
import numpy as np
import datetime
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import argparse


class ForecastModel(ABC):
    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def fit(self, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, time_steps, levels):
        pass


class ARIMA(ForecastModel):
    def description(self):
        return "ARIMA (1, 0, 0)"

    def fit(self, y, **kwargs):

        AR_term = 1
        i_term = 0
        MA_term = 0
        self.model = smARIMA(
            series_df["value"], order=(AR_term, i_term, MA_term)
        )
        self.arima_results = self.model.fit(disp=False)

    def predict(self, time_steps, levels):

        forecast, std_dev, _ = self.arima_results.forecast(steps=time_steps)

        forecast_dict = {"forecast": forecast}

        for level in levels:
            forecast_dict[f"LB_{level}"] = (
                forecast + norm.ppf((1 - level / 100) / 2) * std_dev
            )
            forecast_dict[f"UB_{level}"] = (
                forecast - norm.ppf((1 - level / 100) / 2) * std_dev
            )

        return forecast_dict


class AutoARIMA(ForecastModel):
    def __init__(self):
        # Import the R library
        self.forecast_lib = importr("forecast")
        pandas2ri.activate()

    def description(self):
        # arma = dict(self.fit_results.items())["arma"]

        # AR, MA, SAR, SMA, Period, I, SI
        # 0,   1,   2,   3,      4, 5, 6
        # https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html
        # return f"ARIMA({arma[0]}, {arma[5]}, {arma[1]})({arma[2]}, {arma[6]}, {arma[3]})"

        return self.method

    def fit(self, y, **kwargs):
        self.fit_results = self.forecast_lib.auto_arima(y)

        r_forecast_dict = dict(
            self.forecast_lib.forecast(
                self.fit_results,
                h=1,
                level=robjects.IntVector(kwargs["levels"]),
            ).items()
        )

        self.method = r_forecast_dict["method"][0]

    def predict(self, time_steps, levels):

        r_forecast_dict = dict(
            self.forecast_lib.forecast(
                self.fit_results,
                h=time_steps,
                level=robjects.IntVector(levels),
            ).items()
        )

        forecast_dict = {"forecast": r_forecast_dict["mean"]}

        for i in range(len(levels)):
            forecast_dict[f"LB_{levels[i]}"] = r_forecast_dict["lower"][:, i]
            forecast_dict[f"UB_{levels[i]}"] = r_forecast_dict["upper"][:, i]

        return forecast_dict


def forecast_to_df(
    data_source_dict,
    forecast_dict,
    first_value,
    first_time,
    forecast_len,
    levels,
):

    forecast_df = pd.DataFrame(forecast_dict)

    first_row = {"forecast": first_value}

    for level in levels:
        first_row[f"LB_{level}"] = first_value
        first_row[f"UB_{level}"] = first_value

    final_forecast_df = pd.concat(
        [pd.DataFrame(pd.Series(first_row)).transpose(), forecast_df], axis=0
    )

    final_forecast_df.index = pd.date_range(
        start=first_time,
        periods=forecast_len + 1,
        freq=data_source_dict["frequency"],
    )

    return final_forecast_df


def train_test_split(series, test_steps):

    train_set = series.iloc[:-test_steps]
    test_set = series.iloc[-test_steps:]

    return train_set, test_set


def run_models(sources_path, download_dir_path, forecast_dir_path):

    with open(sources_path) as data_sources_json_file:

        data_sources_list = json.load(data_sources_json_file)

        for data_source_dict in data_sources_list:

            print(data_source_dict["title"])

            # Read local pickle that we created earlier
            f = open(
                f"{download_dir_path}/{data_source_dict['title']}.pkl", "rb"
            )
            downloaded_dict = pickle.load(f)
            f.close()

            series_df = downloaded_dict["series_df"]

            # Hack to align to the end of the quarter
            if data_source_dict["frequency"] == "Q":
                offset = pd.offsets.QuarterEnd()
                series_df.index = series_df.index + offset

            forecast_len = 8
            levels = [50, 75, 95]

            # Form train-validation sets
            train_set, validation_set = train_test_split(
                series_df["value"], forecast_len
            )

            model_class_list = [AutoARIMA]

            metric_list = []

            # Train a whole bunch-o models on the training set
            # and evaluate them on the validation set
            for model_class in model_class_list:

                model = model_class()

                model.fit(train_set, levels=levels)

                model_predictions = model.predict(
                    len(validation_set), levels=levels
                )

                metric_list.append(
                    mean_squared_error(
                        validation_set, model_predictions["forecast"]
                    )
                )

            best_model_class = model_class_list[np.argmin(metric_list)]

            # Retrain best model on full data
            best_model = best_model_class()
            best_model.fit(series_df["value"], levels=levels)

            # Generate final forecast using best model
            forecast_dict = best_model.predict(forecast_len, levels=levels)

            first_value = series_df["value"].iloc[-1]
            first_time = series_df.index[-1]

            forecast_df = forecast_to_df(
                data_source_dict,
                forecast_dict,
                first_value,
                first_time,
                forecast_len,
                levels=levels,
            )

            # Store forecast and related
            data = {
                "data_source_dict": data_source_dict,
                "downloaded_dict": downloaded_dict,
                "forecasted_at": datetime.datetime.now(),
                "forecast_df": forecast_df,
                "model_used": best_model.description(),
            }

            f = open(
                f"{forecast_dir_path}/{data_source_dict['title']}.pkl", "wb"
            )
            pickle.dump(data, f)
            f.close()


if __name__ == "__main__":

    args_dict = {
        "s": {
            "help": "sources file path",
            "default": "../shared_config/data_sources.json",
            "kw": "sources_path",
        },
        "d": {
            "help": "download directory path",
            "default": "../data/downloads",
            "kw": "download_dir_path",
        },
        "f": {
            "help": "forecast directory path",
            "default": "../data/forecasts",
            "kw": "forecast_dir_path",
        },
    }

    parser = argparse.ArgumentParser()

    for k, v in args_dict.items():
        parser.add_argument(f"--{k}", help=v["help"])

    input_args = vars(parser.parse_args())

    final_kwargs = {}
    for k, v in args_dict.items():
        final_kwargs[v["kw"]] = (
            input_args[k] if input_args[k] else v["default"]
        )

    run_models(**final_kwargs)
