import argparse
import datetime
import json
import pickle
import os.path

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import indexable, _num_samples

from models import (
    RNaive,
    RNaive2,
    RAutoARIMA,
    RSimple,
    RHolt,
    RDamped,
    RTheta,
    RComb,
)

pandas2ri.activate()

p_to_use = 1
forecast_len = 8
level = [50, 75, 95]

model_class_list = [
    RNaive,
    # RAutoARIMA,  # RAutoARIMA is very slow!
    RSimple,
    RHolt,
    RDamped,
    RTheta,
    RNaive2,
    RComb,
]


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


class TimeSeriesRollingSplit:
    def __init__(self, h=1, p_to_use=1):

        self.h = h
        self.p_to_use = p_to_use

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        """

        X, y, groups = indexable(X, y, groups)

        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        h = self.h

        min_position = np.maximum(h, int(n_samples * (1 - self.p_to_use)))

        positions = np.flip(np.arange(min_position, n_samples - h))

        for position in positions:

            yield (indices[:position], indices[position : position + h])


def cross_val_score(model, y, cv, scorer, fit_params={}):

    errors = []

    for train_index, test_index in cv.split(y):
        y_train, y_test = y[train_index], y[test_index]

        model.fit(y_train, **fit_params)

        model_predictions = model.predict()

        errors.append(scorer(y_test, model_predictions))

    return np.mean(errors)


# Short-circuit forecasting if hashsums match
def check_cache(download_pickle, cache_pickle):

    # Read local pickle that we created earlier
    f = open(download_pickle, "rb")
    downloaded_dict = pickle.load(f)
    f.close()

    download_hashsum = downloaded_dict["hashsum"]

    # Debug: modify the download_hashsum (as if data had changed)
    # to force re-calculation of all forecasts.
    # download_hashsum = "X" + download_hashsum[1:]

    # print("  - download:", download_hashsum)

    if os.path.isfile(cache_pickle):
        f = open(cache_pickle, "rb")
        cache_dict = pickle.load(f)
        f.close()

        if "hashsum" in cache_dict["downloaded_dict"]:
            cache_hashsum = cache_dict["downloaded_dict"]["hashsum"]
            if cache_hashsum == download_hashsum:
                return downloaded_dict, cache_dict

    return downloaded_dict, None


def run_models(sources_path, download_dir_path, forecast_dir_path):
    with open(sources_path) as data_sources_json_file:

        data_sources_list = json.load(data_sources_json_file)

        for data_source_dict in data_sources_list:

            print(data_source_dict["title"])

            downloaded_dict, cache_dict = check_cache(
                f"{download_dir_path}/{data_source_dict['title']}.pkl",
                f"{forecast_dir_path}/{data_source_dict['title']}.pkl",
            )

            # Read local pickle that we created earlier
            series_df = downloaded_dict["series_df"]

            # Hack to align to the end of the quarter
            if data_source_dict["frequency"] == "Q":
                offset = pd.offsets.QuarterEnd()
                series_df.index = series_df.index + offset

            cv = TimeSeriesRollingSplit(h=forecast_len, p_to_use=p_to_use)
            init_params = {"h": forecast_len, "level": level}
            y = series_df["value"]

            all_forecasts = {}
            forecasted_at = datetime.datetime.now()

            # Train a whole bunch-o models on the training set
            # and evaluate them on the validation set
            for model_class in model_class_list:

                model_name = model_class.name
                if cache_dict and model_name in cache_dict["all_forecasts"]:
                    cached_forecasts = cache_dict["all_forecasts"]
                    all_forecasts[model_name] = cached_forecasts[model_name]
                    print("  - Re-using   :", model_name)
                    continue

                print("  - Calculating:", model_name)

                model = model_class(**init_params)
                cv_score = cross_val_score(model, y, cv, mean_squared_error)

                model.fit(y)
                model_description = model.description()

                # Generate final forecast using best model
                forecast_dict = model.predict_withci()

                first_value = series_df["value"].iloc[-1]
                first_time = series_df.index[-1]

                forecast_df = forecast_to_df(
                    data_source_dict,
                    forecast_dict,
                    first_value,
                    first_time,
                    forecast_len,
                    levels=level,
                )

                all_forecasts[model_name] = {
                    "model_description": model_description,
                    "cv_score": cv_score,
                    "forecast_df": forecast_df,
                }

            # Store forecast and related
            data = {
                "data_source_dict": data_source_dict,
                "downloaded_dict": downloaded_dict,
                "forecasted_at": forecasted_at,
                "all_forecasts": all_forecasts,
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
