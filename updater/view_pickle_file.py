# check file.
import pandas as pd
import pickle
import json
import sys
from slugify import slugify

title = sys.argv[1]

f = open(f"../data/forecasts/{slugify(title)}.pkl", "rb")
downloaded_dict = pickle.load(f)
f.close()

print(downloaded_dict)


f = open(f"../shared_config/data_sources.json", "rb")
data_sources = json.load(f)
f.close()

for data in data_sources:
    # print("\n")
    # print("-"*25)
    title = data["title"]
    # print(title)
    f = open(f"../data/forecasts/{slugify(title)}.pkl", "rb")
    downloaded_dict = pickle.load(f)
    f.close()
    if len(downloaded_dict["all_forecasts"]) < 11:
        print("\n")
        print("-" * 25)
        print(title)
        print(len(downloaded_dict["all_forecasts"]))
