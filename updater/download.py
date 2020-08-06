from abc import ABC
from abc import abstractmethod

import urllib.request as url_request
import urllib.error as url_error

import json
import xml.etree.ElementTree as ET

import pandas as pd
import pickle
import datetime

from multiprocessing.dummy import Pool as ThreadPool

class DataSource(ABC):
    def __init__(self, download_path, title, url, frequency, tags):
        self.download_path = download_path
        self.title = title
        self.url = url
        self.frequency = frequency
        self.tags = tags

    def fetch(self):
        print(self.title)

        series_df = self.download()
        data = {
            "series_df": series_df,
            "downloaded_at": datetime.datetime.now(),
        }

        f = open(f"{self.download_path}/{self.title}.pkl", "wb")
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
        #print(df)
        return df


class Fred(DataSource):

    # Thanks to https://github.com/mortada/fredapi/blob/master/fredapi/fred.py

    def download(self):

        api_key_file = "../shared_config/fred_api_key"
        with open(api_key_file, "r") as kf:
            api_key = kf.readline().strip()

        # To do: api_key should regex match [0-9a-f]{32}
        if not api_key:
            raise ValueError(f"Please add a FRED API key to {api_key_file} .")

        self.url += "&api_key=" + api_key

        try:
            response = url_request.urlopen(self.url)
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
        #print(df)
        return df

supported_data_sources = {
    "AusMacroData": AusMacroData,
    "Fred": Fred
}

def download_data_source(data_source_dict, download_path):

    if "source" not in data_source_dict:
        raise ValueError(f"No source found for {data_source_dict['title']}")

    if data_source_dict["source"] not in supported_data_sources:
        raise ValueError(f"Source {data_source_dict['source']} is not supported")

    # Pop because init of DataSource does not accept source
    source_class = supported_data_sources[data_source_dict.pop("source")]
    data_source_dict["download_path"] = download_path

    source = source_class(**data_source_dict)

    source.fetch()


def download_data(sources_path, download_path):

    with open(sources_path) as data_sources_json_file:

        data_sources_list = json.load(data_sources_json_file)

        pool = ThreadPool(8)

        pool.starmap(download_data_source, [(data_source_dict, download_path) for data_source_dict in data_sources_list])

        pool.close()
        pool.join()

if __name__ == "__main__":
    download_data("../shared_config/data_sources.json", "../data/downloads")
