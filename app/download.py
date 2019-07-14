import json
import pandas as pd
import pickle
import datetime

with open("data_sources.json") as data_sources_json_file:

    data_sources_list = json.load(data_sources_json_file)

    for data_source_dict in data_sources_list:

        print(data_source_dict["title"])

        # Use read_csv to access remote file
        series_df = pd.read_csv(
            data_source_dict["url"],
            usecols=["date", "value"],
            parse_dates=["date"],
            index_col="date",
        )

        data = {
            "series_df": series_df,
            "downloaded_at": datetime.datetime.now(),
        }

        f = open(f"downloads/{data_source_dict['title']}.pkl", "wb")
        pickle.dump(data, f)
        f.close()
