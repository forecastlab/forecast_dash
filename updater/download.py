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
import warnings  # for the world bank FutureWarnings
import re

import os

from slugify import slugify

import traceback


class DataSource(ABC):
    def __init__(
        self,
        download_path,
        title,
        url,
        frequency,
        tags,
        short_title=None,
        data_folder=None,
    ):
        self.download_path = download_path
        self.title = title
        self.short_title = short_title
        self.url = url
        self.frequency = frequency
        self.tags = tags
        self.data_folder = data_folder  # handle incremental data

        # slugify title
        self.filename = slugify(title)

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

            f = open(f"{self.download_path}/{self.filename}.pkl", "wb")
            pickle.dump(data, f)
            f.close()
            state = "OK"
        except Exception as err:
            traceback.print_exc()
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
        with warnings.catch_warnings():  # FutureWarnings Error in the wb package. Issue with loading an empty series.
            warnings.simplefilter(action="ignore", category=FutureWarning)

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


# not sure the best way to handle fuel dataset
# the api is hard to use to access all the data
# the excel files are with different formattings...
class IncrementalData(DataSource):
    @abstractmethod
    def fetch_download_df(self):
        """
        Create a dataframe for the download list

        The dataframe should contain at least following columns:
        `url`, `filename`
        """
        pass

    # functions for get urls to be downloaded
    def _check_new_url(self, url, df, url_col="url", status_col="status"):
        """
        Check if the given url is new to a given dataframe
        """
        url_row = df[df[url_col] == url]
        return True if len(url_row) == 0 else False

    def compare_file_dif(self, new_df=None):
        """
        get the difference between the dataframe from `self.fetch_download_df` and the local pickle file
        """
        if new_df is None:
            raise ValueError("File DataFrame is empty...")

        if not os.path.exists(f"{self.download_path}/{self.data_folder}/"):
            os.makedirs(f"{self.download_path}/{self.data_folder}/")

        download_fname = (
            f"{self.download_path}/{self.data_folder}/download.pkl"
        )
        if not os.path.exists(download_fname):
            new_df["status"] = 0
            return new_df

        old_df = pd.read_pickle(download_fname)

        ### iterate through new_df - check url
        new_url_bool = [
            self._check_new_url(row["url"], old_df)
            for _, row in new_df.iterrows()
        ]

        new_df = new_df.iloc[new_url_bool].copy(deep=True)
        new_df["status"] = 0

        return pd.concat([old_df, new_df], ignore_index=True)

    # download the single file
    def download_single_file(self, url, filename, verbose=True):
        """
        Download a file from a given url to a local file with name `filename`

        Could overwrite this function if there is any data cleaning process
        """
        if verbose:
            print(f"Downloading files from {url}...")
        filepath = f"{self.download_path}/{self.data_folder}/{filename}"
        r = requests.get(url)
        with open(filepath, "wb") as fp:
            fp.write(r.content)

    # download all files and update `download.pkl`
    def download_files(self, file_df):
        """
        Download files from concatenated dataframe
        """
        for i, row in file_df.iterrows():
            url = row["url"]
            status = row["status"]
            filename = row["filename"]

            if status == 0:
                try:
                    self.download_single_file(url, filename)
                    file_df.loc[i, "status"] = 1
                except Exception as err:
                    traceback.print_exc()

        # save download pkl file
        file_df.to_pickle(
            f"{self.download_path}/{self.data_folder}/download.pkl"
        )


class FuelNSWData(IncrementalData):
    def _get_filename_from_url(self, url):
        filename = url.split("/")[-1].split(".")[:-1]
        return ".".join(filename) + ".pkl"

    def fetch_download_df(self):
        fuelfiles_json_url = "https://data.nsw.gov.au/data/api/3/action/package_show?id=a97a46fc-2bdd-4b90-ac7f-0cb1e8d7ac3b"
        fuelfiles_json = requests.get(fuelfiles_json_url).json()
        fuelfiles_df = pd.DataFrame(fuelfiles_json["result"]["resources"])

        fuelfiles_bool = fuelfiles_df["name"].apply(
            lambda x: ("price history" in x.lower())
        )

        fuelfiles_df = fuelfiles_df[fuelfiles_bool].copy(deep=True)

        fuelfiles_name = fuelfiles_df["url"].apply(
            lambda x: self._get_filename_from_url(x)
        )

        return pd.DataFrame(
            {
                "url": fuelfiles_df["url"].to_list(),
                "filename": fuelfiles_name.to_list(),
            }
        )

    # functions to do data cleaning
    def _check_header_row(self, df, key="PriceUpdatedDate", rowlimit=5):
        """find the index of the header row"""
        # check if header is wrong
        header_wrong = False
        for column_name in df.columns:
            if "Unnamed" in column_name:
                header_wrong = True

        if not header_wrong:
            return 0

        for row in range(rowlimit):
            for value in df.iloc[row]:
                if value == key:
                    return row + 1

    def _extract_series(self, excel_path):
        df = pd.read_excel(excel_path).iloc[:, -3:]
        header_row = self._check_header_row(df)
        # check if there are two extra columns
        if header_row > 0:
            df = pd.read_excel(excel_path, header=header_row).iloc[:, -3:]
        return df.dropna().rename(columns={"FuelType": "FuelCode"}), header_row

    def _clean_series(self, excel_path, fueltype="E10"):
        series, header_row = self._extract_series(excel_path)
        series = series.query("FuelCode == @fueltype").reset_index(drop=True)
        if header_row == 2 and pd.api.types.is_object_dtype(
            series["PriceUpdatedDate"]
        ):
            series["PriceUpdatedDate"] = pd.to_datetime(
                series["PriceUpdatedDate"], dayfirst=True
            )
        elif pd.api.types.is_object_dtype(series["PriceUpdatedDate"]):
            series["PriceUpdatedDate"] = pd.to_datetime(
                series["PriceUpdatedDate"]
            )

        # add month
        series_date = series["PriceUpdatedDate"].iloc[0]
        series["PriceMonth"] = pd.to_datetime(
            "{}-{:0>2}".format(series_date.year, series_date.month)
        )
        # add week
        refer_date = pd.to_datetime("2022-09-05")
        week_delta = (series["PriceUpdatedDate"] - refer_date).apply(
            lambda x: x.days
        ) // 7
        series["PriceWeek"] = refer_date + week_delta * pd.Timedelta(
            7, unit="days"
        )
        return series

    # rewrite download_single_file function
    def download_single_file(self, url, filename, verbose=True):
        if verbose:
            print(f"Downloading files from {url}...")
        filepath = f"{self.download_path}/{self.data_folder}/{filename}"
        series_df = self._clean_series(url)
        return series_df.to_pickle(filepath)

    def download(self):
        ### download files
        file_df = self.fetch_download_df()
        file_df = self.compare_file_dif(new_df=file_df)
        self.download_files(file_df)

        series_list = []
        for _, row in file_df.iterrows():
            if row["status"] == 1:
                filepath = f"{self.download_path}/{self.data_folder}/{row['filename']}"
                series_list.append(pd.read_pickle(filepath))

        df = pd.concat(series_list)
        if self.frequency == "M":  # month?
            df = df.groupby("PriceMonth").median()[["Price"]]
        else:  # week
            df = df.groupby("PriceWeek").median()[["Price"]]

        df.index.name = "date"

        return df.rename(columns={"Price": "value"})


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
                "FuelNSW": FuelNSWData,
            }

            source_class = all_source_classes[data_source_dict.pop("source")]
            data_source_dict["download_path"] = download_path

            source = source_class(**data_source_dict)

            source.fetch()


if __name__ == "__main__":
    # download_data("../shared_config/data_sources.json", "../data/downloads")
    download_data(
        "../shared_config/testing_data_sources.json", "../data/downloads"
    )
