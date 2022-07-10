import datetime
import json
import pickle
from abc import ABC
from abc import abstractmethod
from hashlib import sha256

import numpy as np
import pandas as pd
import requests
from requests.exceptions import HTTPError


import wbgapi as wb  # for world bank data
import re


class DataSource(ABC):
    def __init__(
        self, download_path, title, url, frequency, tags, short_title=None
    ):
        self.download_path = download_path
        self.title = title
        self.short_title = short_title
        self.url = url
        self.frequency = frequency
        self.tags = tags

    def fetch(self):

        try:
            series_df = self.download()
            self.hashsum = sha256(series_df.to_csv().encode()).hexdigest()
            # print("  -", hashsum)
            data_version = self.data_versioning()

            data = {
                "hashsum": self.hashsum,
                "series_df": series_df,
                "downloaded_at": datetime.datetime.now(),
                "data_version": data_version,
                "frequency": self.frequency,
            }

            f = open(f"{self.download_path}/{self.title}.pkl", "wb")
            pickle.dump(data, f)
            f.close()
            state = "OK"
        except Exception as e:
            print(
                f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}"
            )
            state = "FAILED"
        finally:
            print(f"{self.title} - {state}")

    def data_versioning(self):
        try:
            f = open(f"{self.download_path}/{self.title}.pkl", "rb")
            previous_download = pickle.load(f)
            f.close()

            if not self.hashsum == previous_download["hashsum"]:
                if "data_version" in previous_download:
                    data_version += 1
                    print(f"{self.title} - Version Updated")
                else:
                    data_version = previous_download["data_version"]
        except:
            data_version = 100000
        finally:
            print(f"{self.title} - {data_version}")

        return data_version

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

        payload = {"api_key": api_key, "file_type": "json"}

        try:
            response = requests.get(self.url, params=payload)

            # Raise exception if response fails
            # (response.status_code outside the 200 to 400 range).
            response.raise_for_status()

        except HTTPError as http_err:
            raise ValueError(f"HTTP error: {http_err} .")

        data = response.json()

        df = pd.DataFrame(data["observations"])[["date", "value"]]

        # Set index
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        # FRED uses '.' to represent null values - replace with NaN
        df["value"] = df["value"].replace(".", np.NaN).astype(float)
        df = df.dropna()

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

        # This will raise an exception if JSON decoding fails
        json_data = response.json()

        dates = []
        values = []

        if self.frequency == "MS":
            dates = [j["date"] for j in json_data["months"]]
            values = [float(j["value"]) for j in json_data["months"]]

        elif self.frequency == "Q":
            # Reformat "YYYY Qn" to "YYYY-Qn" before passing to pd.to_datetime
            dates = [
                "-".join(j["date"].split()) for j in json_data["quarters"]
            ]
            values = [float(j["value"]) for j in json_data["quarters"]]

        df = pd.DataFrame(
            values, index=pd.to_datetime(dates), columns=["value"]
        )
        df.index.name = "date"

        # print(df)
        return df


class WorldBankData(DataSource):
    def download(self):
        """
        indicator: The series indicator. Full list is here = https://data.worldbank.org/indicator?tab=all
        region: The region of interest.
        """
        url_cut = re.sub("https://api.worldbank.org/v2/", "", self.url)

        indicator = url_cut.split("/")[3]
        region = url_cut.split("/")[1].split(";")

        df = wb.data.DataFrame(
            indicator, region, numericTimeKeys=True, skipBlanks=True
        )
        df = df.transpose()
        df.dropna(
            how="all", inplace=True
        )  # Sometimes the last date has not been entered yet.
        df = pd.DataFrame(
            df[region].values,
            index=pd.to_datetime(df.index, format="%Y"),
            columns=["value"],
        )

        return df


class ABSData(DataSource):
    """
    The API data guide is avaialble here: https://www.abs.gov.au/about/data-services/application-programming-interfaces-apis/data-api-user-guide

    The basic syntax is  /data/{dataflowIdentifier}/{dataKey}
    dataflowIdentifier: The necessary dataflow canbe found from this website: https://api.data.abs.gov.au/dataflow
    dataKey: The data key takes paramters for the search. Usually have to have a inital one and the rest can be separated out by '.'. The parameters to use can be found from https://api.data.abs.gov.au/datastructure/ABS/{dataflowIdentifier}. Use https://api.data.abs.gov.au/datastructure/ABS/{dataflowIdentifier}?references=codelist to find what values to use. The first one usually starts with 'M' and is the Measure.

    More complex queries can be used.
    """

    def download(self):

        df = pd.read_csv(self.url)

        df = pd.DataFrame(
            df["OBS_VALUE"].values,
            index=pd.to_datetime(df["TIME_PERIOD"]),
            columns=["value"],
        )
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
                "WorldBank": WorldBankData,
                "ABS": ABSData,
            }

            source_class = all_source_classes[data_source_dict.pop("source")]
            data_source_dict["download_path"] = download_path

            source = source_class(**data_source_dict)

            source.fetch()


if __name__ == "__main__":
    download_data("../shared_config/data_sources.json", "../data/downloads")
    # download_data("../shared_config/testing_data_sources.json", "../data/downloads")
