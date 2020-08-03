from abc import ABC
from abc import abstractmethod

import urllib.request as url_request
import urllib.parse as url_parse
import urllib.error as url_error

import json
import xml.etree.ElementTree as et

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

        f = open(
            f"{self.download_dir_path}/{self.title}.pkl", "wb"
        )
        pickle.dump(data, f)
        f.close()

    @abstractmethod
    def download(self):
        pass 
    
    
class AusMacroData(DataSource):

    def download(self):

        assert( not self.url_opts )
        
        # Use read_csv to access remote file
        data = pd.read_csv(
            self.url,
            usecols=["date", "value"],
            parse_dates=["date"],
            index_col="date",
        )
        print(data)
        return data
            
        
class Fred(DataSource):

    def parse(self, date_str, format='%Y-%m-%d'):

        rv = pd.to_datetime(date_str, format=format)
        if hasattr(rv, 'to_pydatetime'):
            rv = rv.to_pydatetime()
        return rv
    
    def download(self):

        api_key_file = "../shared_config/fred_api_key"
        with open(api_key_file, 'r') as kf:
            api_key = kf.readline().strip()

        print("api_key:" + api_key)
        self.url_opts += "&api_key=" + api_key

        try:
            response = url_request.urlopen(self.url + self.url_opts)
            root = et.fromstring(response.read())
        except url_error.HTTPError as exc:
            root = et.fromstring(exc.read())
            raise ValueError(root.get('message'))

        if root is None:
            raise ValueError("Failed to retrieve any data.")

        data = {}
        for child in root:
            val = float(child.get('value'))
            data[self.parse(child.get('date'))] = val

        data = pd.Series(data)
        print( data )
        return data

    
def download_data(sources_path):

    with open(sources_path) as data_sources_json_file:

        data_sources_list = json.load(data_sources_json_file)

        for data_source_dict in data_sources_list:

            all_source_classes = { "AusMacroData": AusMacroData,
                                   "Fred": Fred }

            source_class = all_source_classes[data_source_dict.pop("source")]
            source = source_class(**data_source_dict)

            source.fetch()

if __name__ == "__main__":
    download_data("../shared_config/data_sources.json")
