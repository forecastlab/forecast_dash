# check file.
import pandas as pd
import pickle
import json
import sys
from slugify import slugify
import numpy as np

f = open(f"../shared_config/data_sources.json", "rb")
data_sources = json.load(f)
f.close()


def select_best_model(data_dict, CV_score_function="MSE"):
    # use the MSE as the default scoring function for identifying the best model.
    # Extract ( model_name, cv_score ) for each model.
    all_models = []
    all_cv_scores = []
    for model_name, forecast_dict in data_dict["all_forecasts"].items():
        if forecast_dict:
            all_models.append(model_name)
            if (
                forecast_dict["state"] == "OK"
                and type(forecast_dict["cv_score"]) == dict
            ):
                all_cv_scores.append(
                    forecast_dict["cv_score"][CV_score_function]
                )
            else:
                all_cv_scores.append(forecast_dict["cv_score"])
    # Select the best model.
    model_name = all_models[np.argmin(all_cv_scores)]
    return model_name


searchable_details = {}
for data in data_sources:
    searchable_details[data["title"]] = [data["title"]]
    title = data['title']
    try:
        [data["short_title"]] = [data["title"]]
    except:
        pass
    for tag in data["tags"]:
        if tag in searchable_details.keys():
            searchable_details[tag] += [data["title"]]
        else:
            searchable_details[tag] = [data["title"]]
    f = open(f"../data/forecasts/{slugify(title)}.pkl", "rb")
    downloaded_dict = pickle.load(f)
    f.close()
    best_method = select_best_model(downloaded_dict)
    print(best_method)
    if best_method in searchable_details.keys():
        searchable_details[best_method] += [data["title"]]
    else:
        searchable_details[best_method] = [data["title"]]

with open('../shared_config/search_a_series.json', 'w') as searches_json_file:
     searches_json_file.write(json.dumps(searchable_details))