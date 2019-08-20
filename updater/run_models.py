import argparse
import datetime
import json
import pickle
from abc import ABC
from abc import abstractmethod

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.metrics import mean_squared_error

forecast_len = 8
levels = [50, 75, 95]


class ForecastModel(ABC):
    @property
    @staticmethod
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def fit(self, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, time_steps, levels):
        pass


class RForecastModel(ForecastModel, ABC):
    @property
    @staticmethod
    @abstractmethod
    def forecast_model_name():
        pass

    @property
    @staticmethod
    @abstractmethod
    def forecast_model_params():
        pass

    def __init__(self):
        # Import the R library
        self.forecast_lib = importr("forecast")
        pandas2ri.activate()

    def description(self):
        return self.method

    def fit(self, y, **kwargs):

        self.fit_results = getattr(
            self.forecast_lib, type(self).forecast_model_name
        )(y, **type(self).forecast_model_params)

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


class RSimple(RForecastModel):
    name = "Simple Exponential Smoothing (ZNN)"

    forecast_model_name = "ets"

    forecast_model_params = {"model": "ZNN"}


class RHolt(RForecastModel):
    name = "Holt-Winters (ZNN)"

    forecast_model_name = "ets"

    forecast_model_params = {"model": "ZZN"}


class RDamped(RForecastModel):
    name = "Damped (ZZN, Damped)"

    forecast_model_name = "ets"

    forecast_model_params = {"model": "ZZN", "damped": True}


class RTheta(RForecastModel):
    name = "Theta"

    forecast_model_name = "thetaf"

    forecast_model_params = {"level": robjects.IntVector(levels)}


class RAutoARIMA(RForecastModel):
    name = "Auto ARIMA"

    forecast_model_name = "auto_arima"

    forecast_model_params = {}


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

            # Form train-validation sets
            train_set, validation_set = train_test_split(
                series_df["value"], forecast_len
            )

            model_class_list = [RAutoARIMA, RSimple, RHolt, RDamped, RTheta]

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
                "model_name": best_model.name,
                "model_description": best_model.description(),
            }

            f = open(
                f"{forecast_dir_path}/{data_source_dict['title']}.pkl", "wb"
            )
            pickle.dump(data, f)
            f.close()

        # Save statistics
        print("Generating Statistics")
        data = {"models_used": [m.name for m in model_class_list]}

        f = open(f"{forecast_dir_path}/statistics.pkl", "wb")
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
