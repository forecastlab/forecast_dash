# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:17:36 2022

@author: ArdiMirzaei
"""

import json
import requests
import pandas as pd
import os

with open("./shared_config/data_sources.json") as data_sources_json_file:
    data_sources_list = json.load(data_sources_json_file)

#%%

results = []
for data_source in data_sources_list:
    paylod = None
    title = data_source["title"]
    url = data_source["url"]
    if data_source["source"] == "Fred":
        FRED_API = os.environ["FRED_API_KEY"]
        payload = {"api_key": FRED_API, "file_type": "json"}
    try:
        status = requests.get(
            url, timeout=(6.05, 12), params=payload
        ).status_code
    except requests.Timeout:
        status = "Timeout"
    except Exception as e:
        print(e)

    print(f"{title} - {status}")
    results.append([title, url, status])

results = pd.DataFrame(results)

#%%
results.columns = ["Title", "URL", "Status"]
results.replace({"Status": {"Timeout": 99}})
#%%
results_grouped = results["Status"].value_counts().reset_index()

print(
    f"There are {results_grouped['Status'][results_grouped['index'] == 99].sum()} that were not reached because they timed out"
)
print(
    f"There are { results_grouped['Status'][(results_grouped['index'] >= 200) & (results_grouped['index'] <= 400)].sum()} data sources that are reachable."
)
print(
    f"The following {results_grouped['Status'][results_grouped['index'] > 400].sum()} are not reachable:\n"
)
print(results[["Title", "Status"]][results["Status"] > 400])
