import argparse
import datetime
import json
import os.path
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import indexable, _num_samples
from statsmodels.tsa.tsatools import freq_to_period

import importlib

from slugify import slugify

# number of forecasts to make for series with different frequencies
# monthly data (freq = 12): 18 forecasts
# quarterly data (freq = 4): 8 forecasts
# weekly data (freq = 52): 13 forecasts
# annual data (freq = 1): 4 forecasts
forecast_len_map = {52: 13, 12: 18, 4: 8, 1: 4}
default_forecast_len = 8
p_to_use = 1
level = [50, 75, 95]

model_str_list = [
    "RNaive",
    "RAutoARIMA",
    "RSimple",
    "RHolt",
    "RDamped",
    "RTheta",
    "RNaive2",
    "RComb",
    "LinearRegressionForecast",
    "RNN_M4_benchmark",
    "FBProphet",
]

# import model classes
models_module = __import__("models")
model_class_list = [
    getattr(models_module, model_str) for model_str in model_str_list
]


class ScoringFunctions:
    """
    Scoring functions to use in the CV function
    """

    def __init__(self, y_train, y_true, y_pred):
        self.y_train = y_train
        self.y_true = y_true
        self.y_pred = y_pred
        self.error = self.y_true - self.y_pred

    def mean_squared_error(self):
        """
        mean squared error of forecasts
        """

        MSE = np.mean(np.square(self.error))
        return MSE

    def mean_absolute_scaled_error(self, period=1):
        """
        mean absolute scaled error of forecasts.
        """

        y_pred_naive = self.y_train[:-period]
        denominator = mean_absolute_error(self.y_train[period:], y_pred_naive)
        if denominator < 1e-8:
            denominator = 1e-8
        MASE = np.mean(np.abs(self.error / denominator))
        return MASE

    def Winkler_score(self, upper_ci, lower_ci, alpha):
        """
        Winkler score for evaluating the quality of prediction intervals.
        see e.g., https://otexts.com/fpp3/distaccuracy.html
        """
        y1_mask = (self.y_true < lower_ci).astype(int)
        y2_mask = (
            (self.y_true >= lower_ci) & (self.y_true <= upper_ci)
        ).astype(int)
        y3_mask = (self.y_true > upper_ci).astype(int)

        w1 = (
            (upper_ci - lower_ci) + (2 / alpha) * (lower_ci - self.y_true)
        ) * y1_mask
        w2 = (upper_ci - lower_ci) * y2_mask
        w3 = (
            (upper_ci - lower_ci) + (2 / alpha) * (self.y_true - upper_ci)
        ) * y3_mask

        W = (
            np.sum(w1) + np.sum(w2) + np.sum(w3)
        )  # Sum of individual all h-step Winkler scores

        return W

    def weighted_quantile_loss(self, ytau, tau):
        """
        The Weighted Quantile Loss (wQL) metric measures the accuracy of a model at a specified quantile.
        see e.g., https://docs.aws.amazon.com/forecast/latest/dg/metrics.html
        The wQL metric cannot be calculated for the mean forecast.
        """
        lq = np.maximum(self.y_true - ytau, np.zeros(self.y_true.shape)) * tau
        uq = np.maximum(ytau - self.y_true, np.zeros(self.y_true.shape)) * (
            1 - tau
        )
        return 2 * (lq.sum() + uq.sum()) / np.abs(self.y_true).sum()

    def weighted_absolute_percentage_error(self):
        """
        The Weighted Absolute Percentage Error (WAPE) measures the overall deviation of forecasted values from observed values.
        see: https://docs.aws.amazon.com/forecast/latest/dg/metrics.html
        """
        return np.abs(self.error).sum() / np.abs(self.y_true).sum()

    def mean_absolute_percentage_error(self):
        """
        mean value of the percentage error between observed and predicted values
        """
        return np.abs(1 - self.y_pred / self.y_true).mean()

    def symmetric_mean_absolute_percentage_error(self):
        """
        Symmetric mean absolute percentage error defined as
        ``(y - y_pred).abs() / (y.abs() + y_pred.abs())``
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        denominator = np.maximum(
            np.abs(self.y_pred) + np.abs(self.y_true),
            np.ones(self.y_true.shape) * 1e-8,
        )
        loss = (
            np.abs(self.y_pred - self.y_true) / denominator
        ).sum() / self.y_true.shape[0]

        return loss


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


def cross_val_score(model, y, cv, fit_params={}):

    errors = {
        score: []
        for score in ["MSE", "MASE"]
        + [f"{x}% Winkler" for x in level]
        + ["wQL25", "WAPE", "SMAPE"]
    }  # list of scores for each scoring function

    for train_index, test_index in cv.split(y):
        y_train, y_test = y[train_index], y[test_index]

        model.fit(y_train, **fit_params)

        model_predictions = model.predict()

        # Compute score for each scoring function
        sf = ScoringFunctions(
            y_train=y_train, y_true=y_test, y_pred=model_predictions
        )
        errors["MSE"].append(sf.mean_squared_error())
        try:  # if the model instance has a periodicity attribute use that
            if (
                y_train.shape[0] > model.period
            ):  # can only compute the MASE when there is at least 1 lagged observation of the same period
                model_period = model.period
                errors["MASE"].append(
                    sf.mean_absolute_scaled_error(period=model_period)
                )
        except AttributeError:  # otherwise try to compute the periodicity of the training data
            freq = getattr(y_train.index, "inferred_freq", None)
            period = freq_to_period(freq)
            errors["MASE"].append(sf.mean_absolute_scaled_error(period=period))

        # Scores for 95% prediction intervals
        forecast_dict_index = model.predict_withci()
        for CI_alpha in model.level:
            lower_ci = forecast_dict_index[f"LB_{CI_alpha}"]
            upper_ci = forecast_dict_index[f"UB_{CI_alpha}"]
            errors[f"{CI_alpha}% Winkler"].append(
                sf.Winkler_score(
                    upper_ci=upper_ci,
                    lower_ci=lower_ci,
                    alpha=1 - (CI_alpha / 100),
                )
            )

        # Scores for wQL50
        # errors["wQL50"].append(
        #     sf.weighted_quantile_loss(model_predictions, 0.5)
        # )

        # Scores for wQL25
        errors["wQL25"].append(
            sf.weighted_quantile_loss(forecast_dict_index["LB_50"], 0.25)
        )

        # Scores for WAPE
        errors["WAPE"].append(sf.weighted_absolute_percentage_error())

        # Scores for SMAPE
        errors["SMAPE"].append(sf.symmetric_mean_absolute_percentage_error())

    mean_errors = {key: np.mean(value) for key, value in errors.items()}
    return mean_errors


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


def run_job(job_dict, cv, model_params):

    try:
        series_df = job_dict["downloaded_dict"]["series_df"]

        y = series_df["value"]

        model = job_dict["model_cls"](**model_params)

        cv_score = cross_val_score(model, y, cv)

        model.fit(y)

        forecast_dict = model.predict_withci()

        first_value = series_df["value"].iloc[-1]
        first_time = series_df.index[-1]

        forecast_df = forecast_to_df(
            job_dict["data_source_dict"],
            forecast_dict,
            first_value,
            first_time,
            model_params["h"],
            levels=level,
        )

        result = {
            "state": "OK",
            "model_description": model.description(),
            "cv_score": cv_score,
            "forecast_df": forecast_df,
        }

    except:
        result = {"state": "FAIL"}

    # ### Following block was used for debug
    # except Exception as e:
    #     print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    #     result = {"state": "FAIL"}

    print(f"{job_dict['title']} - {job_dict['model_cls']} - {result['state']}")

    return job_dict, result


def run_models(sources_path, download_dir_path, forecast_dir_path):

    # Save statistics
    print("Generating Statistics")
    data = {"models_used": [m.name for m in model_class_list]}

    f = open(f"{forecast_dir_path}/statistics.pkl", "wb")
    pickle.dump(data, f)
    f.close()

    with open(sources_path) as data_sources_json_file:

        data_sources_list = json.load(data_sources_json_file)

    # Results storage
    series_dict = {}

    job_list = []
    cv_instance_list = []
    model_params_list = []

    # Parse JSON and cache
    for data_source_dict in data_sources_list:

        # print(data_source_dict["title"])
        data_filename = slugify(data_source_dict["title"])

        try:
            downloaded_dict, cache_dict = check_cache(
                f"{download_dir_path}/{data_filename}.pkl",
                f"{forecast_dir_path}/{data_filename}.pkl",
            )

            # Read local pickle that we created earlier
            series_df = downloaded_dict["series_df"]

            # Hack to align to the end of the quarter
            if data_source_dict["frequency"] == "Q":
                offset = pd.offsets.QuarterEnd()
                series_df.index = series_df.index + offset

            all_forecasts = {}

            # Find period of the series and define the TimeSeriesRollingSplit instance
            series_freq = getattr(series_df.index, "inferred_freq", None)
            if series_freq is not None:
                series_period = freq_to_period(series_freq)
                series_h = forecast_len_map[series_period]
            else:
                series_period = None
                series_h = default_forecast_len

            series_cv = TimeSeriesRollingSplit(
                h=series_h, p_to_use=p_to_use
            )  # seperate TimeSeriesRollingSplit instance for each series to account for differences in frequencies

            for model_class in model_class_list:

                model_name = model_class.name

                # Use cached results
                if cache_dict and model_name in cache_dict["all_forecasts"]:
                    cached_forecasts = cache_dict["all_forecasts"]
                    result = cached_forecasts[model_name]

                else:
                    # Add to job list
                    job_list.append(
                        {
                            "title": data_source_dict["title"],
                            "model_cls": model_class,
                            "data_source_dict": data_source_dict,
                            "downloaded_dict": downloaded_dict,
                        }
                    )
                    # Add CV instance to list
                    cv_instance_list.append(series_cv)
                    # Add model parameters dictionary to list
                    model_params_list.append(
                        {
                            "h": series_h,
                            "level": level,
                            "period": series_period,
                        }
                    )

                    # Temporarily set result to empty
                    result = {}

                all_forecasts[model_name] = result

            series_dict[data_source_dict["title"]] = {
                "data_source_dict": data_source_dict,
                "downloaded_dict": downloaded_dict,
                "forecasted_at": datetime.datetime.now(),
                "all_forecasts": all_forecasts,
            }
        except:
            print(f"FAILED to setup jobs for: '{data_source_dict['title']}'")

    pool = Pool(cpu_count())
    results = pool.starmap(
        run_job,
        [
            [job_list[i], cv_instance_list[i], model_params_list[i]]
            for i in range(len(job_list))
        ],
    )

    # Insert results of jobs into dictionary
    for result in results:
        model_name = result[0]["model_cls"].name
        series_title = result[0]["title"]

        if result[1]["state"] == "OK":
            series_dict[series_title]["all_forecasts"][model_name] = result[1]
        else:
            series_dict[series_title]["all_forecasts"].pop(model_name, None)

    # Write all series pickles to disk
    for series_title, series_data in series_dict.items():
        data_filename = slugify(series_data["data_source_dict"]["title"])

        f = open(
            f"{forecast_dir_path}/{data_filename}.pkl",
            "wb",
        )
        pickle.dump(series_data, f)
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
