from abc import ABC
from abc import abstractmethod

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()


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

        # Import the R library
        self.forecast_lib = importr("forecast")

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


# Some methods in R-forecast produce immediate forecasts
# https://otexts.com/fpp2/the-forecast-package-in-r.html
# e.g. naive, rwf, holt
class RDirectForecastModel(RModel):
    def get_r_forecast_dict(self):
        return dict(
            getattr(self.forecast_lib, type(self).r_forecast_model_name)(
                y=self.y, h=self.h, level=self.r_level
            ).items()
        )

    def fit(self, y):

        self.y = y

        super().fit(y)


class RForecastModel(RModel):
    def get_r_forecast_dict(self):
        return dict(
            self.forecast_lib.forecast(
                self.fit_results, h=self.h, level=self.r_level
            ).items()
        )

    def fit(self, y):

        fit_params = {"y": y, **type(self).forecast_model_params}

        self.fit_results = getattr(
            self.forecast_lib, type(self).r_forecast_model_name
        )(**fit_params)

        super().fit(y)


class RNaive(RDirectForecastModel):
    name = "Naive"

    r_forecast_model_name = "naive"


class RTheta(RDirectForecastModel):
    name = "Theta"

    r_forecast_model_name = "thetaf"


class RSimple(RForecastModel):
    name = "Simple Exponential Smoothing (ZNN)"

    r_forecast_model_name = "ets"

    forecast_model_params = {"model": "ZNN"}


class RHolt(RForecastModel):
    name = "Holt-Winters (ZNN)"

    r_forecast_model_name = "ets"

    forecast_model_params = {"model": "ZZN"}


class RDamped(RForecastModel):
    name = "Damped (ZZN, Damped)"

    r_forecast_model_name = "ets"

    forecast_model_params = {"model": "ZZN", "damped": True}


class RAutoARIMA(RForecastModel):
    name = "Auto ARIMA"

    r_forecast_model_name = "auto_arima"
