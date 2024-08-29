import pandas as pd
import numpy as np

from abc import ABC
from abc import abstractmethod

from scipy import stats

from statsmodels.tsa.tsatools import freq_to_period

from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPRegressor

import torch

# from prophet import Prophet

import warnings


### functions for the MLP and RNN models ###
def split_into_train(y, lags):
    """
    creates a training dataset contains lags of the original time series. The original time series is truncated to match the length
    of the training dataset.
    """
    X_train = pd.DataFrame(
        columns=[x for x in range(lags, 0, -1)]
    )  # initialize a dataframe to hold lags of the time series
    for i in range(lags, 0, -1):
        X_train[i] = y.shift(i)
    X_train.dropna(inplace=True)  # remove all missing lagged data
    y_train = y.iloc[lags:]  # truncate y to match the lagged x_train matrix
    return X_train, y_train


def detrend(data):
    """
    Calculates a & b parameters of a linear regression line used to remove trends from the time series
    """
    x = np.arange(len(data))
    a, b = np.polyfit(
        x, data, 1
    )  # coefficients are returned with the highest power first
    return a, b


def seasonality_test(original_ts, ppy):
    """
    Seasonality test
    """
    s = acf(original_ts, 1)
    for i in range(2, ppy):
        s = s + (acf(original_ts, i) ** 2)

    limit = 1.645 * (np.sqrt((1 + 2 * (s**2)) / len(original_ts)))

    return (abs(acf(original_ts, ppy))) > limit


def acf(data, k):
    """
    Autocorrelation function
    """
    m = np.mean(data)
    s1 = 0
    for i in range(k, len(data)):
        s1 = s1 + ((data[i] - m) * (data[i - k] - m))

    s2 = 0
    for i in range(0, len(data)):
        s2 = s2 + ((data[i] - m) ** 2)

    return float(s1 / s2)


def remove_seasonality(y, h, period):
    """
    Compute (additive) seasonality component of y and the h-step shead forecasts of the seasoanlity component.
    This is a slightly adapted version of the statsmodels STL decomposition/forecast code.
    We use a simple linear regression trend line for removing the trend, not local linear regression.
    """
    # compute trend using simple linear regression
    a, b = detrend(y.values)
    X = np.column_stack([np.ones(len(y)), np.arange(0, len(y), 1)])
    beta = np.array([b, a])
    trend = X @ beta

    # compute trend using local linear regression
    # trend = sm.nonparametric.lowess(endog = y, exog = np.array([x for x in range(len(y))]), frac = 0.3, delta = 0, return_sorted = False)

    detrended = y - trend  # remove estimated trend from y

    period_averages = np.array(
        [np.nanmean(detrended[i::period]) for i in range(period)]
    )  # select and average every period'th point in the time series
    # 0-center the period avgs
    period_averages -= np.mean(period_averages)
    seasonal = np.tile(period_averages, len(y) // period + 1)[
        : len(y)
    ]  # Repeat the seasonal array to the time series length and truncate to match the observed series

    # Find the seasonal predictions
    seasonal_ix = 0
    max_correlation = -np.inf
    detrended_array = np.asanyarray(y - trend).squeeze()

    for i, x in enumerate(period_averages):
        # work slices backward from end of detrended observations
        if i == 0:
            # slicing w/ [x:-0] doesn't work
            detrended_slice = detrended_array[-len(period_averages) :]
        else:
            detrended_slice = detrended_array[-(len(period_averages) + i) : -i]
        # calculate corr b/w period_avgs and detrend_slice
        this_correlation = np.correlate(detrended_slice, period_averages)[0]
        if this_correlation > max_correlation:
            # update ix and max correlation
            max_correlation = this_correlation
            seasonal_ix = i

    # roll seasonal signal to matching phase
    rolled_period_averages = np.roll(period_averages, -seasonal_ix)
    # tile as many time as needed to reach "h", then truncate
    seasonal_forecasts = np.tile(
        rolled_period_averages, (h // len(period_averages) + 1)
    )[:h]

    return trend, seasonal, seasonal_forecasts


class ForecastModel(BaseEstimator, ABC):
    def __init__(self, h=None, level=None, period=None):
        self.h = h
        self.level = level
        self.period = period

    @property
    @staticmethod
    @abstractmethod
    def name():
        pass

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def fit(self, y):
        pass

    @abstractmethod
    def predict(self):
        pass


class LinearRegressionForecast(ForecastModel):
    """
    Linear regression model from the M4 forecasting competition benchmarks
    """

    name = "Linear Regression"
    method = "Linear Regression"

    def __init__(self, h=1, level=[], period=None):
        self.input_size = 3  # number of inputs to the forecasting model (lags of the time series)
        super().__init__(h, level, period)

    def description(self):
        return self.method

    def fit(self, y):

        self.y = y.dropna()
        self.y_tilde = self.y.copy()  # detrended and de-seasonalized copy of y
        # detrending
        self.a, self.b = detrend(self.y.values)
        for i in range(len(self.y)):
            self.y_tilde.iloc[i] = self.y_tilde.iloc[i] - (
                (self.a * i) + self.b
            )

        self.seasonal_bool = False
        if self.period is not None:
            if (
                (seasonality_test(self.y, self.period))
                and ~(self.period is None)
                and (len(self.y) > 2 * self.period)
            ):
                self.seasonal_bool = True
                # remove seasonality and compute the required h-step ahead seasonal components
                trend, seasonal, self.seasonal_forecasts = remove_seasonality(
                    self.y, self.h, self.period
                )
                self.y_tilde = self.y_tilde - seasonal
        # create X data matrix as 3 lags of y
        X_train, y_train = split_into_train(self.y_tilde, lags=self.input_size)

        # Convert to np.array to match input dtype requirement for Sklearn MLP model
        X_train = X_train.values
        y_train = y_train.values

        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()

        self.model.fit(X_train, y_train)

        self.resids = (
            self.model.predict(X_train) - y_train
        )  # training sample residuals

    def predict(self):
        self.y_tilde_hat = []
        X_test = self.y_tilde.iloc[-self.input_size :].values.reshape(
            1, self.input_size
        )
        for i in range(
            self.h
        ):  # make h-step ahead predictions of detrended and de-seasonalized data
            MLP_forecast = self.model.predict(X_test)[0]
            # assign previous forecast as new X_test point
            X_test[0, :] = np.roll(X_test[0], -1)
            X_test[0, -1] = MLP_forecast
            self.y_tilde_hat.append(MLP_forecast)

        # Add back the trend and seasonality to forecasts
        y_hat = np.array(self.y_tilde_hat).copy()
        for i in range(self.h):
            y_hat[i] = y_hat[i] + ((self.a * (len(self.y) + i + 1)) + self.b)
        if self.seasonal_bool == True:
            y_hat = y_hat + self.seasonal_forecasts

        return y_hat

    def predict_withci(self):
        y_hat = (
            self.predict()
        )  # call predict to get the correct self.yhat, self.y_tilde_hat after CV loss estimation
        forecast_dict = {"forecast": y_hat}

        sigma_hat_resid = np.std(
            self.resids
        )  # standard deviation of (de-trended and de-seasonalized) residuals
        sigma_h = (
            np.array([np.sqrt(i) for i in range(1, self.h + 1)])
            * sigma_hat_resid
        )  # standard deviation of h-step ahead forecast (naive sqrt of time approach)

        for i in range(len(self.level)):
            lower_CI_tilde = (
                self.y_tilde_hat
                - stats.norm.ppf(self.level[i] / 100) * sigma_h
            )
            upper_CI_tilde = (
                self.y_tilde_hat
                + stats.norm.ppf(self.level[i] / 100) * sigma_h
            )

            # Add back trend and seasonality to the CI
            lower_CI = lower_CI_tilde.copy()
            upper_CI = upper_CI_tilde.copy()
            for j in range(self.h):
                lower_CI[j] = lower_CI[j] + (
                    (self.a * (len(self.y) + j + 1)) + self.b
                )
                upper_CI[j] = upper_CI[j] + (
                    (self.a * (len(self.y) + j + 1)) + self.b
                )
            if self.seasonal_bool == True:
                lower_CI = lower_CI + self.seasonal_forecasts
                upper_CI = upper_CI + self.seasonal_forecasts

            forecast_dict[f"LB_{self.level[i]}"] = lower_CI
            forecast_dict[f"UB_{self.level[i]}"] = upper_CI

        return forecast_dict


class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(SimpleRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.rnn = torch.nn.RNN(
            input_size, hidden_dim, n_layers, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # Initializing hidden state for first input using method defined below
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class RNN_M4_benchmark(ForecastModel):
    """
    Recurrent neural network model from the M4 forecasting competition benchmarks
    """

    name = "RNN"
    method = "RNN M4 Competition Benchmark"

    def __init__(self, h=1, level=[], period=None):
        self.input_size = 3  # number of inputs to the forecasting model (lags of the time series)
        super().__init__(h, level, period)

    def description(self):
        return self.method

    def fit(self, y, **kwargs):
        self.y = y
        self.y_tilde = y.copy()  # detrended and de-seasonalized y

        # Find period of y
        freq = getattr(self.y.index, "inferred_freq", None)
        self.period = freq_to_period(freq)

        # detrending
        a, b = detrend(self.y.values)
        self.a = a
        self.b = b
        for i in range(len(y)):
            self.y_tilde.iloc[i] = self.y_tilde.iloc[i] - (
                (self.a * i) + self.b
            )

        self.seasonal_bool = False
        if self.period is not None:
            if (
                (seasonality_test(self.y, self.period))
                and ~(self.period is None)
                and (len(self.y) > 2 * self.period)
            ):
                self.seasonal_bool = True
                # remove seasonality and compute the required h-step ahead seasonal components
                trend, seasonal, self.seasonal_forecasts = remove_seasonality(
                    self.y, self.h, self.period
                )
                self.y_tilde = self.y_tilde - seasonal

        # create X data matrix as input_size lags of y
        X_train, y_train = split_into_train(self.y_tilde, lags=self.input_size)

        # convert to np.array to match input dtype requirement for Keras RNN model
        X_train = X_train.values

        y_train = y_train.values
        # reshape to match expected input
        X_train = np.reshape(X_train, (-1, 1, self.input_size))

        X_train = torch.tensor(X_train, dtype=torch.float)
        y_train = torch.tensor(y_train, dtype=torch.float)

        # Create the training dataset
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        # Create a data loader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=X_train.shape[0], shuffle=True
        )

        self.model = SimpleRNN(self.input_size, 1, 6, 1)

        # Specify loss
        loss_function = torch.nn.MSELoss()

        # Specify optimizer and parameters
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        max_iter = 100
        loss = []

        for epoch in range(max_iter):
            # Get a batch of data from data loader
            for X_batch, y_batch in dataloader:
                # Reset the gradients
                optimizer.zero_grad()
                # Forward pass
                predictions = self.model(X_batch).squeeze()
                # Calculate loss
                l = loss_function(predictions, y_batch)
                # Compute gradients
                l.backward()
                # Update using gradients
                optimizer.step()

                loss.append(l.item())

        with torch.no_grad():
            predictions = self.model(X_train)

        self.resids = (
            predictions - y_train
        ).numpy()  # training sample residuals

    def predict(self):
        self.y_tilde_hat = []
        X_test = self.y_tilde.iloc[-self.input_size :].values

        X_test_np = X_test.reshape(-1, self.input_size, 1)

        X_test_torch = np.reshape(X_test, (-1, 1, self.input_size))
        X_test_torch = torch.tensor(X_test_torch, dtype=torch.float)

        for i in range(
            self.h
        ):  # make h-step ahead predictions of detrended and de-seasonalized data
            with torch.no_grad():
                # BUG HEREEEEE
                RNN_forecast = self.model(X_test_torch).numpy()[0]

            # RNN_forecast = self.model.predict(X_test)[0]
            # assign previous forecast as new X_test point
            X_test_np[0, :] = np.roll(X_test_np[0], -1)
            X_test_np[0, -1] = RNN_forecast
            self.y_tilde_hat.append(RNN_forecast)

        y_hat = np.array(self.y_tilde_hat).copy()
        self.y_tilde_hat = np.array(self.y_tilde_hat).flatten()
        y_hat = y_hat.flatten()

        # add back the trend and seasonality to the forecasts
        for i in range(self.h):
            y_hat[i] = y_hat[i] + ((self.a * (len(self.y) + i + 1)) + self.b)
        if self.seasonal_bool == True:
            y_hat = y_hat + self.seasonal_forecasts

        return y_hat

    def predict_withci(self):
        y_hat = (
            self.predict()
        )  # call predict to get the correct self.yhat, self.y_tilde_hat after CV loss estimation
        forecast_dict = {"forecast": y_hat}

        sigma_hat_resid = np.std(
            self.resids
        )  # standard deviation of (de-trended and de-seasonalized) residuals
        sigma_h = (
            np.array([np.sqrt(i) for i in range(1, self.h + 1)])
            * sigma_hat_resid
        )  # standard deviation of h-step ahead forecast (naive sqrt of time approach)

        for i in range(len(self.level)):
            lower_CI_tilde = (
                self.y_tilde_hat
                - stats.norm.ppf(self.level[i] / 100) * sigma_h
            )
            upper_CI_tilde = (
                self.y_tilde_hat
                + stats.norm.ppf(self.level[i] / 100) * sigma_h
            )

            # Add back trend and seasonality to the CI
            lower_CI = lower_CI_tilde.copy()
            upper_CI = upper_CI_tilde.copy()
            for j in range(self.h):
                lower_CI[j] = lower_CI[j] + (
                    (self.a * (len(self.y) + j + 1)) + self.b
                )
                upper_CI[j] = upper_CI[j] + (
                    (self.a * (len(self.y) + j + 1)) + self.b
                )
            if self.seasonal_bool == True:
                lower_CI = lower_CI + self.seasonal_forecasts
                upper_CI = upper_CI + self.seasonal_forecasts

            forecast_dict[f"LB_{self.level[i]}"] = lower_CI
            forecast_dict[f"UB_{self.level[i]}"] = upper_CI

        return forecast_dict


class RModel(ForecastModel, ABC):
    @property
    @staticmethod
    @abstractmethod
    def r_forecast_model_name():
        pass

    forecast_model_params = {}

    def __init__(self, h=1, level=[], period=None):
        super().__init__(h, level, period)

        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri

        pandas2ri.activate()

        self.r_level = robjects.IntVector(self.level)

        # Import the R sources
        if type(self).r_forecast_lib.endswith(".R"):
            self.forecast_lib = robjects.r.source(type(self).r_forecast_lib)
            self.forecast_func = getattr(
                robjects.r, type(self).r_forecast_model_name
            )
        else:
            # Import the R library
            from rpy2.robjects.packages import importr

            self.forecast_lib = importr(type(self).r_forecast_lib)
            self.forecast_func = getattr(
                self.forecast_lib, type(self).r_forecast_model_name
            )

            self.r_generics_lib = importr(
                "generics"
            )  # as of forecast 8.17, generics pacakges does the forecasting.

    def description(self):
        return self.method

    @abstractmethod
    def get_r_forecast_dict(self):
        pass

    def fit(self, y):
        r_forecast_dict = self.get_r_forecast_dict()

        self.method = r_forecast_dict["method"][0]

    def predict(self):
        r_forecast_dict = self.get_r_forecast_dict()

        prediction = r_forecast_dict["mean"]

        return prediction

    def predict_withci(self):
        r_forecast_dict = self.get_r_forecast_dict()

        forecast_dict = {"forecast": r_forecast_dict["mean"]}

        for i in range(len(self.level)):
            forecast_dict[f"LB_{self.level[i]}"] = r_forecast_dict["lower"][
                :, i
            ]
            forecast_dict[f"UB_{self.level[i]}"] = r_forecast_dict["upper"][
                :, i
            ]

        return forecast_dict


# Some methods in R-forecast produce immediate forecasts: the R call is
# <model_name>( y, <model_params> ) .
# Examples are forecast::naive, rwf, holt .
# https://otexts.com/fpp2/the-forecast-package-in-r.html
class RDirectForecastModel(RModel):
    def get_r_forecast_dict(self):
        return dict(
            self.forecast_func(y=self.y, h=self.h, level=self.r_level).items()
        )

    def fit(self, y):
        self.y = y

        super().fit(y)


# Models for which the R call is forecast( <model_name>( y, <model_params> ) ) .
# Examples are forecast::ets and forecast::auto.arima .
class RForecastModel(RModel):
    def get_r_forecast_dict(self):
        return dict(
            self.r_generics_lib.forecast(
                self.fit_results, h=self.h, level=self.r_level
            ).items()
        )

    def fit(self, y):
        fit_params = {"y": y, **type(self).forecast_model_params}

        self.fit_results = self.forecast_func(**fit_params)

        super().fit(y)


class RSmoothForecastModel(RModel):
    def get_r_forecast_dict(self):
        r_level = [
            i / 100 for i in self.r_level
        ]  # CES reads levels as decimals

        return dict(
            self.forecast_func(y=self.y, h=self.h, level=r_level).items()
        )

    def fit(self, y):
        self.y = y

        r_forecast_dict = self.get_r_forecast_dict()

        self.method = "CES"


class RNaive(RDirectForecastModel):
    name = "Naive"

    r_forecast_lib = "forecast"

    r_forecast_model_name = "naive"


class RNaive2(RDirectForecastModel):
    name = "Seasonally Adjusted Naive"

    r_forecast_lib = "seasadj.R"

    r_forecast_model_name = "naive2"


class RTheta(RDirectForecastModel):
    name = "Theta"

    r_forecast_lib = "forecast"

    r_forecast_model_name = "thetaf"


class RSimple(RForecastModel):
    name = "Simple Exponential Smoothing (ZNN)"

    r_forecast_lib = "forecast"

    r_forecast_model_name = "ets"

    forecast_model_params = {"model": "ZNN"}


class RHolt(RForecastModel):
    name = "Holt-Winters (ZNN)"

    r_forecast_lib = "forecast"

    r_forecast_model_name = "ets"

    forecast_model_params = {"model": "ZZN"}


class RDamped(RForecastModel):
    name = "Damped (ZZN, Damped)"

    r_forecast_lib = "forecast"

    r_forecast_model_name = "ets"

    forecast_model_params = {"model": "ZZN", "damped": True}


class RAutoARIMA(RForecastModel):
    name = "Auto ARIMA"

    r_forecast_lib = "forecast"

    r_forecast_model_name = "auto_arima"


class RComb(RDirectForecastModel):
    name = "Combination M4 Benchmark"

    r_forecast_lib = "seasadj.R"

    r_forecast_model_name = "comb"


# # prophet model
# class FBProphet(ForecastModel):
#     name = "facebook-prophet"
#
#     method = "facebook-prophet"
#
#     def fit(self, y):
#         # rename two column as `date` and `y` per prophet requirement
#         self.y = y.reset_index().rename(columns={"date": "ds", "value": "y"})
#
#         self.model = Prophet().fit(self.y)
#
#     def _make_future(self):
#         """make future dataframe for prediction"""
#         return self.model.make_future_dataframe(periods=self.h).tail(self.h)
#
#     def predict(self):
#         # make future prediction dataframe
#         future = self._make_future()
#         return self.model.predict(future)["yhat"].to_numpy()
#
#     def predict_withci(self):
#         future = self._make_future()
#         forecast_dict = {"forecast": self.predict()}
#
#         for i in range(len(self.level)):
#             self.model.interval_width = self.level[i] * 0.01
#             forecast_df = self.model.predict(future)
#
#             # add intervals
#             forecast_dict[f"LB_{self.level[i]}"] = forecast_df[
#                 "yhat_lower"
#             ].to_numpy()
#             forecast_dict[f"UB_{self.level[i]}"] = forecast_df[
#                 "yhat_upper"
#             ].to_numpy()
#
#         return forecast_dict
#
#     def description(self):
#         return self.model


class RCES(RSmoothForecastModel):
    name = "Complex Exponential Smoothing"

    r_forecast_lib = "smooth_package.R"

    r_forecast_model_name = "complex_es"
