import json
import pandas as pd

with open("data_sources.json") as data_sources_json_file:

    data_sources_list = json.load(data_sources_json_file)

    for data_source_dict in data_sources_list:

        print(data_source_dict["title"])

        # Use read_csv to access remote file
        df = pd.read_csv(data_source_dict['url'], usecols=['date', 'value'])

        df.to_csv(f"downloads/{data_source_dict['title']}.csv", index = False)