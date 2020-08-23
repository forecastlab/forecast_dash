from abc import ABC
from abc import abstractmethod

from sklearn.base import BaseEstimator


class ForecastModel(BaseEstimator, ABC):
    def __init__(self, h=None, level=None):
        self.set_params(**{"h": h, "level": level})

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


class RModel(ForecastModel, ABC):
    @property
    @staticmethod
    @abstractmethod
    def r_forecast_model_name():
        pass

    forecast_model_params = {}

    def __init__(self, h=1, level=[]):

        super().__init__(h, level)

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
