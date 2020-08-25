from abc import ABC
from abc import abstractmethod

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import RMSprop

def split_into_train(y):
    train = np.array(y)
    x_train, y_train = train[:-1], np.roll(train, -3)[:-3]

    # reshape input to be [samples, time steps, features] (N-NF samples, 1 time step, 1 feature)
    x_train = x_train.reshape(-1, 1)
    temp_train = np.roll(x_train, -1)
        
    for x in range(1, 3):
        x_train = np.concatenate((x_train[:-1], temp_train[:-1]), 1)
        temp_train = np.roll(temp_train, -1)[:-1]

    return x_train, y_train


class ForecastModel(ABC):
    def __init__(self, h, level):
        self.h = h
        self.level = level

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

    def get_params(self, deep=True):
        return {}


class RNNModel(ForecastModel, ABC):

    method = "SimpleRNN"
    name = "RNN M4 Benchmark"

    def __init__(self, h, level):

        super().__init__(h, level)


    def name(self):
        return self.name

    def description(self):
        return self.method

    def fit(self, y):

        x_train, y_train = split_into_train(y)

        # reshape to match expected input
        x_train = np.reshape(x_train, (-1, 3, 1))

        self.model = Sequential(
            [
                SimpleRNN(
                    6,
                    input_shape=(3, 1),
                    activation="linear",
                    use_bias=False,
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                    bias_initializer="zeros",
                    dropout=0.0,
                    recurrent_dropout=0.0,
                ),
                Dense(1, use_bias=True, activation="linear"),
            ]
        )
        opt = RMSprop(lr=0.001)
        self.model.compile(loss="mean_squared_error", optimizer=opt)

        # fit the model to the training data
        self.model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=0)

        return self


    def predict(self, y):
        y_hat = []
        x_test = np.array(y[-3:]).reshape(-1,1)
        x_test = np.reshape(x_test, (-1, 3, 1))
        last_prediction = self.model.predict(x_test)[0]
        for i in range(0, self.h):
            y_hat.append(last_prediction)
            x_test[0] = np.roll(x_test[0], -1)
            x_test[0, (len(x_test[0]) - 1)] = last_prediction
            last_prediction = self.model.predict(x_test)[0]
        print(y_hat)
        return np.reshape(y_hat, (-1))


class RModel(ForecastModel, ABC):
    @property
    @staticmethod
    @abstractmethod
    def r_forecast_model_name():
        pass

    forecast_model_params = {}

    def __init__(self, h, level):

        super().__init__(h, level)

        self.r_level = robjects.IntVector(level)

        # Import the R sources
        if type(self).r_forecast_lib.endswith(".R"):
            self.forecast_lib = robjects.r.source(type(self).r_forecast_lib)
            self.forecast_func = getattr(
                robjects.r, type(self).r_forecast_model_name
            )
        else:
            # Import the R library
            self.forecast_lib = importr(type(self).r_forecast_lib)
            self.forecast_func = getattr(
                self.forecast_lib, type(self).r_forecast_model_name
            )

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
            self.forecast_lib.forecast(
                self.fit_results, h=self.h, level=self.r_level
            ).items()
        )

    def fit(self, y):

        fit_params = {"y": y, **type(self).forecast_model_params}

        self.fit_results = self.forecast_func(**fit_params)

        super().fit(y)


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
