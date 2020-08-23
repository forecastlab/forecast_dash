import time

from models import RNaive, RNaive2, RAutoARIMA, RSimple, RHolt, RDamped, RTheta
from rpy2.robjects import pandas2ri

pandas2ri.activate()
import json
import pickle
from run_models import (
    TimeSeriesRollingSplit,
    cross_val_score,
    mean_squared_error,
)
import pandas as pd

sources_path = "../shared_config/data_sources.json"
download_dir_path = "../data/downloads"

data_sources_json_file = open(sources_path)

data_sources_list = json.load(data_sources_json_file)

data_source_dict = data_sources_list[0]

print(data_source_dict["title"])

# Read local pickle that we created earlier
f = open(f"{download_dir_path}/{data_source_dict['title']}.pkl", "rb")
downloaded_dict = pickle.load(f)
f.close()


model_class_list = [
    RNaive,
    RAutoARIMA,  # RAutoARIMA is very slow!
    RSimple,
    RHolt,
    RDamped,
    RTheta,
    RNaive2,
]

# model_dict = {
#     model_class.__name__: model_class()
#     for model_class in model_class_list
# }


def run_model_data(model_cls, data_dict, cv, model_params):

    series_df = data_dict["series_df"]

    # Hack to align to the end of the quarter
    if data_source_dict["frequency"] == "Q":
        offset = pd.offsets.QuarterEnd()
        series_df.index = series_df.index + offset

    y = series_df["value"]

    model = model_cls(**model_params)

    cv_score = cross_val_score(model, y, cv, mean_squared_error)

    return cv_score


from multiprocessing import Pool, Manager, cpu_count


pool = Pool(cpu_count())
# pool = Pool(1)
# print( model_class_list[:1])

forecast_len = 8
level = [50, 75, 95]
p_to_use = 1
cv = TimeSeriesRollingSplit(h=forecast_len, p_to_use=p_to_use)
model_params = {"h": forecast_len, "level": level}

results = pool.starmap(
    run_model_data,
    [
        [model_cls, downloaded_dict, cv, model_params]
        for model_cls in model_class_list
    ],
)

print(results)


# 1. Collect list of model and forecasts

# 2. Run them in parallel

# 3. Process results


# for models in model_class_list:
#     model = model_class(**init_params)
#
#
#
#
#
# model_class = RAutoARIMA
# s = time.time()
# model = model_class(**init_params)
# print(time.time() - s)
#
#
#
#
# # def f(x):
# #     return x*x
# #
# # if __name__ == '__main__':
# #     with Pool(5) as p:
# #         print(p.starmap(f, [[1], [2], [3]]))
#
# def run_model(y, cv):
#     # return 0
#     s = time.time()
#     cv_score = cross_val_score(model, y, cv, mean_squared_error)
#     print(time.time() - s)
#     return cv_score
#
#
# # run_model(model, y, cv)
#
# pool = Pool(cpu_count())
# results = pool.starmap(run_model, [[y, cv] for i in range(3)])
