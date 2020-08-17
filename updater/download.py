from abc import ABC
from abc import abstractmethod

import requests
from requests.exceptions import HTTPError

import json
import xml.etree.ElementTree as ET

import pandas as pd
import pickle
import datetime
from hashlib import sha256


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

        hashsum = sha256(series_df.to_csv().encode()).hexdigest()
        #print("  -", hashsum)

        data = {
            "hashsum": hashsum,
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
        # print(df)
        return df


class Fred(DataSource):

    # Thanks to https://github.com/mortada/fredapi/blob/master/fredapi/fred.py
    # and https://realpython.com/python-requests/

    def download(self):

        api_key_file = "../shared_config/fred_api_key"
        with open(api_key_file, "r") as kf:
            api_key = kf.readline().strip()

        # To do: api_key should regex match [0-9a-f]{32}
        if not api_key:
            raise ValueError(f"Please add a FRED API key to {api_key_file} .")

        payload = {"api_key": api_key}

        try:
            response = requests.get(self.url, params=payload)

            # Raise exception if response fails
            # (response.status_code outside the 200 to 400 range).
            response.raise_for_status()

        except HTTPError as http_err:
            raise ValueError(f"HTTP error: {http_err} .")

        root = ET.fromstring(response.text)
        if not root:
            raise ValueError("Failed to retrieve any data.")

        dates = []
        values = []
        for child in root:
            dates.append(pd.to_datetime(child.get("date"), format="%Y-%m-%d"))
            values.append(float(child.get("value")))

        df = pd.DataFrame(values, index=dates, columns=["value"])
        df.index.name = "date"

        return df


class Ons(DataSource):
    def download(self):

        try:
            # ONS currently rejects requests that use the default User-Agent
            # (python-urllib/3.x.y). Set the header manually to pretend to be
            # a 'real' browser.
            response = requests.get(
                self.url, headers={"User-Agent": "Mozilla/5.0"}
            )

            # Raise exception if response fails
            # (response.status_code outside the 200 to 400 range).
            response.raise_for_status()

        except HTTPError as http_err:
            raise ValueError(f"HTTP error: {http_err} .")

        # This will raise an exception JSON decoding fails
        data = response.json()

        dates = []
        values = []
        for el in data["months"]:
            dates.append(pd.to_datetime(el["date"], format="%Y %b"))
            values.append(float(el["value"]))

        df = pd.DataFrame(values, index=dates, columns=["value"])
        df.index.name = "date"

        return df


def download_data(sources_path, download_path):

    with open(sources_path) as data_sources_json_file:

        data_sources_list = json.load(data_sources_json_file)

        for data_source_dict in data_sources_list:

            all_source_classes = {
                "AusMacroData": AusMacroData,
                "Fred": Fred,
                "Ons": Ons,
            }

            source_class = all_source_classes[data_source_dict.pop("source")]
            data_source_dict["download_path"] = download_path

            source = source_class(**data_source_dict)

            source.fetch()


if __name__ == "__main__":
    download_data("../shared_config/data_sources.json", "../data/downloads")
