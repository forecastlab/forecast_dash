# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:17:36 2022

@author: ArdiMirzaei
"""

import json
import requests
import pandas as pd

with open("./shared_config/data_sources.json") as data_sources_json_file:
    data_sources_list = json.load(data_sources_json_file)

#%%

results = []
for data_source in data_sources_list:
    title = data_source["title"]
    url = data_source["url"]
    status = requests.get(url).status_code
    results.append([title, url, status])

results = pd.DataFrame(results)

#%%
results.columns = ["Title", "URL", "Status"]
#%%
results_grouped = results["Status"].value_counts().reset_index()

print(
    f"There are { results_grouped['Status'][(results_grouped['index'] >= 200) & (results_grouped['index'] <= 400)].sum()} data sources that are reachable."
)
print(
    f"The following {results_grouped['Status'][results_grouped['index'] > 400].sum()} are not reachable:\n"
)
print(results[["Title", "Status"]][results["Status"] > 400])
