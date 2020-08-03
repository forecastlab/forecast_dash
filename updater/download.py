from abc import ABC
from abc import abstractmethod

import urllib.request as url_request
import urllib.error as url_error

import json
import xml.etree.ElementTree as ET

import pandas as pd
import pickle
import datetime


class DataSource(ABC):

    download_dir_path = "../data/downloads"

    def __init__(self, title, url, url_opts, frequency, tags):
        self.title = title
        self.url = url
        self.url_opts = url_opts
        self.frequency = frequency
        self.tags = tags

    def fetch(self):
        print(self.title)

        series_df = self.download()
        data = {
            "series_df": series_df,
            "downloaded_at": datetime.datetime.now(),
        }

        f = open(f"{self.download_dir_path}/{self.title}.pkl", "wb")
        pickle.dump(data, f)
        f.close()

    @abstractmethod
    def download(self):
        pass


class AusMacroData(DataSource):
    def download(self):
        # Use read_csv to access remote file
        df = pd.read_csv(
            self.url,
            usecols=["date", "value"],
            parse_dates=["date"],
            index_col="date",
        )
        print(df)
        return df


class Fred(DataSource):

    # Based on https://github.com/mortada/fredapi/blob/master/fredapi/fred.py

    api_key_file = "../shared_config/fred_api_key"
    with open(api_key_file, "r") as kf:
        api_key = kf.readline().strip()
        
    if not api_key:
        raise ValueError( f"Please add a FRED API key to {api_key_file} ." ) 
       
    def download(self):

        self.url_opts += "&api_key=" + self.api_key

        try:
            response = url_request.urlopen(self.url + self.url_opts)
            root = ET.fromstring(response.read())
        except url_error.HTTPError as exc:
            root = ET.fromstring(exc.read())
            raise ValueError(root.get("message"))

        if root is None:
            raise ValueError("Failed to retrieve any data.")

        dates = []
        values = []
        for child in root:
            dates.append(pd.to_datetime(child.get("date"), format="%Y-%m-%d"))
            values.append(float(child.get("value")))

        df = pd.DataFrame(values, index=dates, columns=["value"])
        df.index.name = "date"
        print(df)
        return df


def download_data(sources_path):

    with open(sources_path) as data_sources_json_file:

        data_sources_list = json.load(data_sources_json_file)

        for data_source_dict in data_sources_list:

            all_source_classes = {"AusMacroData": AusMacroData, "Fred": Fred}

            source_class = all_source_classes[data_source_dict.pop("source")]
            source = source_class(**data_source_dict)

            source.fetch()


if __name__ == "__main__":
    download_data("../shared_config/data_sources.json")
